package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const (
	arxivAPIURL    = "https://export.arxiv.org/api/query"
	ar5ivBaseURL   = "https://ar5iv.labs.arxiv.org/html/"
	deepseekAPIURL = "https://api.deepseek.com/chat/completions"
)

// enableFullPaperSummary controls whether the director fetches the full paper
// text from ar5iv and summarizes it with DeepSeek, instead of using the abstract.
var enableFullPaperSummary = true

// experimentName is set at build time via -ldflags.
var experimentName string

//go:embed configs/*.json
var configFS embed.FS

// httpClient forces IPv4 and has a generous timeout for slow reasoning models.
var httpClient = &http.Client{
	Timeout: 10 * time.Minute,
	Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp4", addr)
		},
	},
}

// apiLogger is set in main() and used by callDeepSeek for API logging.
var apiLogger *log.Logger

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type directorConfig struct {
	Name             string   `json:"name"`
	DeepSeekModel    string   `json:"deepseek_model"`
	Temperature      float64  `json:"temperature"`
	ArxivCategories  []string `json:"arxiv_categories"`
	ArxivSearchTerms []string `json:"arxiv_search_terms"`
	SystemPrompt     string   `json:"system_prompt"`
	UserPrompt       string   `json:"user_prompt"`
}

func loadConfig(logger *log.Logger) directorConfig {
	if experimentName == "" {
		logger.Fatal("no experiment name set (binary must be built with -ldflags '-X main.experimentName=...')")
	}

	path := "configs/" + experimentName + ".json"
	data, err := configFS.ReadFile(path)
	if err != nil {
		logger.Fatalf("config not found for experiment %q: %v", experimentName, err)
	}

	var cfg directorConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		logger.Fatalf("parsing config %s: %v", path, err)
	}

	logger.Printf("loaded config: %s (model=%s, temp=%.1f, %d categories, %d search terms)",
		cfg.Name, cfg.DeepSeekModel, cfg.Temperature, len(cfg.ArxivCategories), len(cfg.ArxivSearchTerms))
	return cfg
}

// ---------------------------------------------------------------------------
// arXiv types & fetch
// ---------------------------------------------------------------------------

type arxivFeed struct {
	XMLName xml.Name     `xml:"feed"`
	Entries []arxivEntry `xml:"entry"`
}

type arxivEntry struct {
	Title   string        `xml:"title"`
	Summary string        `xml:"summary"`
	Authors []arxivAuthor `xml:"author"`
	Links   []arxivLink   `xml:"link"`
}

type arxivAuthor struct {
	Name string `xml:"name"`
}

type arxivLink struct {
	Href string `xml:"href,attr"`
	Type string `xml:"type,attr"`
}

func fetchArxivByQuery(logger *log.Logger, query string, offset int) (*arxivEntry, error) {
	reqURL := fmt.Sprintf("%s?search_query=%s&start=%d&max_results=1&sortBy=relevance&sortOrder=descending",
		arxivAPIURL, url.QueryEscape(query), offset)

	resp, err := doGet(reqURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var feed arxivFeed
	if err := xml.NewDecoder(resp.Body).Decode(&feed); err != nil {
		return nil, err
	}
	if len(feed.Entries) == 0 {
		return nil, nil
	}
	return &feed.Entries[0], nil
}

func fetchRandomArxiv(logger *log.Logger, cfg directorConfig) (string, string, *arxivEntry, error) {
	term := cfg.ArxivSearchTerms[rand.IntN(len(cfg.ArxivSearchTerms))]
	cat := cfg.ArxivCategories[rand.IntN(len(cfg.ArxivCategories))]
	offset := rand.IntN(50)

	// Try 1: term + category (most relevant)
	query := fmt.Sprintf("cat:%s AND all:\"%s\"", cat, term)
	logger.Printf("arxiv query: %q offset=%d", query, offset)
	entry, err := fetchArxivByQuery(logger, query, offset)
	if err != nil {
		logger.Printf("arxiv attempt 1 failed: %v", err)
	}

	// Try 2: term only, no category filter (broader)
	if entry == nil {
		query = fmt.Sprintf("all:\"%s\"", term)
		offset = rand.IntN(100)
		logger.Printf("arxiv fallback to term-only: %q offset=%d", query, offset)
		entry, err = fetchArxivByQuery(logger, query, offset)
		if err != nil {
			logger.Printf("arxiv attempt 2 failed: %v", err)
		}
	}

	// Try 3: different term, no category (last resort)
	if entry == nil {
		term = cfg.ArxivSearchTerms[rand.IntN(len(cfg.ArxivSearchTerms))]
		query = fmt.Sprintf("all:\"%s\"", term)
		logger.Printf("arxiv last resort: %q", query)
		entry, err = fetchArxivByQuery(logger, query, 0)
		if err != nil {
			return "", "", nil, fmt.Errorf("all arxiv attempts failed: %w", err)
		}
		if entry == nil {
			return "", "", nil, fmt.Errorf("no arxiv entries found after 3 attempts")
		}
	}

	title := strings.Join(strings.Fields(entry.Title), " ")
	abstract := strings.Join(strings.Fields(entry.Summary), " ")
	logger.Printf("got paper: %q", title)
	return title, abstract, entry, nil
}

// ---------------------------------------------------------------------------
// ar5iv full paper fetch & summarization
// ---------------------------------------------------------------------------

// extractArxivID extracts the paper ID from an arxiv entry's links.
// e.g. "https://arxiv.org/abs/2302.13971v1" → "2302.13971"
func extractArxivID(entry *arxivEntry) string {
	for _, link := range entry.Links {
		href := link.Href
		if strings.Contains(href, "arxiv.org/abs/") {
			parts := strings.Split(href, "/abs/")
			if len(parts) == 2 {
				id := parts[1]
				// strip version suffix (e.g. "v1", "v2")
				if idx := strings.Index(id, "v"); idx > 0 {
					id = id[:idx]
				}
				return id
			}
		}
	}
	return ""
}

// fetchFullPaperText fetches the HTML version of a paper from ar5iv and
// extracts the plain text content from the <article> element.
func fetchFullPaperText(logger *log.Logger, paperID string) (string, error) {
	reqURL := ar5ivBaseURL + paperID
	logger.Printf("fetching full paper from %s", reqURL)

	resp, err := doGet(reqURL)
	if err != nil {
		return "", fmt.Errorf("fetching ar5iv: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("reading ar5iv response: %w", err)
	}

	html := string(body)

	// Extract content within <article...>...</article>
	articleStart := strings.Index(html, "<article")
	articleEnd := strings.Index(html, "</article>")
	if articleStart >= 0 && articleEnd > articleStart {
		html = html[articleStart : articleEnd+len("</article>")]
	}

	// Strip HTML tags
	var text strings.Builder
	inTag := false
	for _, r := range html {
		if r == '<' {
			inTag = true
			continue
		}
		if r == '>' {
			inTag = false
			text.WriteRune(' ')
			continue
		}
		if !inTag {
			text.WriteRune(r)
		}
	}

	// Collapse whitespace and trim
	lines := strings.Split(text.String(), "\n")
	var cleaned []string
	for _, line := range lines {
		line = strings.Join(strings.Fields(line), " ")
		if line != "" {
			cleaned = append(cleaned, line)
		}
	}
	result := strings.Join(cleaned, "\n")

	// Truncate to ~200K chars (~100K tokens) to keep costs reasonable
	const maxChars = 200000
	if len(result) > maxChars {
		result = result[:maxChars] + "\n\n[truncated]"
		logger.Printf("paper text truncated to %d chars", maxChars)
	}

	logger.Printf("extracted %d chars of paper text", len(result))
	return result, nil
}

// summarizePaper calls DeepSeek to summarize the full paper text into
// 2-3 paragraphs focused on the main idea and novelty.
func summarizePaper(logger *log.Logger, cfg directorConfig, apiKey, paperText string) (string, error) {
	logger.Println("summarizing full paper with deepseek-reasoner...")

	systemPrompt := `You are an expert ML researcher extracting actionable techniques from papers.
Summarize this paper in 2-3 short paragraphs with this structure:
1. TECHNIQUE: What specific method/trick/modification did they introduce? Describe the concrete mechanism (e.g. "they replaced X with Y", "they added Z to the loss function", "they scaled W by factor K"). Include formulas, hyperparameter values, or algorithmic steps if present.
2. IMPLEMENTATION: How is it implemented? What layers/modules/functions are affected? What changes to a standard transformer/training loop would be needed to replicate this?
3. RESULTS: What quantitative improvements did they achieve? On what scale (model size, dataset, compute)?

Be concrete and specific. Avoid narrative framing ("this paper addresses...", "the key novelty is..."). Go straight to what they did and how.`
	userPrompt := paperText

	summary, err := callDeepSeek(cfg, apiKey, "deepseek-reasoner", systemPrompt, userPrompt)
	if err != nil {
		return "", fmt.Errorf("paper summarization failed: %w", err)
	}

	logger.Printf("paper summary generated (%d chars)", len(summary))
	return summary, nil
}

// ---------------------------------------------------------------------------
// Paper prefetch cache
// ---------------------------------------------------------------------------

const prefetchCacheFile = ".director_prefetch.json"

type prefetchCache struct {
	PaperTitle    string `json:"paper_title"`
	PaperAbstract string `json:"paper_abstract"`
	PaperSummary  string `json:"paper_summary"`
	Timestamp     string `json:"timestamp"`
}

func prefetchCachePath() string {
	return prefetchCacheFile
}

// loadPrefetchCache atomically claims the cache by renaming before reading,
// preventing two concurrent director processes from using the same cache.
func loadPrefetchCache(logger *log.Logger) *prefetchCache {
	claimedPath := prefetchCachePath() + ".claimed"
	if err := os.Rename(prefetchCachePath(), claimedPath); err != nil {
		return nil
	}

	data, err := os.ReadFile(claimedPath)
	os.Remove(claimedPath)
	if err != nil {
		return nil
	}

	var cache prefetchCache
	if err := json.Unmarshal(data, &cache); err != nil {
		logger.Printf("WARNING: corrupt prefetch cache, ignoring: %v", err)
		return nil
	}

	logger.Printf("loaded prefetch cache (paper: %q)", cache.PaperTitle)
	return &cache
}

func savePrefetchCache(logger *log.Logger, cache *prefetchCache) error {
	cache.Timestamp = time.Now().UTC().Format(time.RFC3339)
	data, err := json.Marshal(cache)
	if err != nil {
		return fmt.Errorf("marshaling prefetch cache: %w", err)
	}
	if err := os.WriteFile(prefetchCachePath(), data, 0o644); err != nil {
		return fmt.Errorf("writing prefetch cache: %w", err)
	}
	logger.Printf("saved prefetch cache (paper: %q)", cache.PaperTitle)
	return nil
}

// runPrefetch fetches a random paper, summarizes it, and saves to cache.
func runPrefetch(logger *log.Logger, cfg directorConfig, apiKey string) {
	logger.Println("prefetch: starting paper fetch and summarization...")

	paperTitle, paperAbstract, arxivEntry, err := fetchRandomArxiv(logger, cfg)
	if err != nil {
		logger.Printf("prefetch: arxiv fetch failed: %v", err)
		return
	}

	paperSummary := ""
	if enableFullPaperSummary && arxivEntry != nil {
		paperID := extractArxivID(arxivEntry)
		if paperID != "" {
			fullText, fetchErr := fetchFullPaperText(logger, paperID)
			if fetchErr != nil {
				logger.Printf("prefetch: full paper fetch failed: %v", fetchErr)
			} else {
				pSum, sumErr := summarizePaper(logger, cfg, apiKey, fullText)
				if sumErr != nil {
					logger.Printf("prefetch: paper summarization failed: %v", sumErr)
				} else {
					paperSummary = pSum
				}
			}
		}
	}

	cache := &prefetchCache{
		PaperTitle:    paperTitle,
		PaperAbstract: paperAbstract,
		PaperSummary:  paperSummary,
	}
	if err := savePrefetchCache(logger, cache); err != nil {
		logger.Printf("prefetch: failed to save cache: %v", err)
	}

	logger.Println("prefetch: done")
}

// spawnPrefetch launches a detached child process to prefetch the next paper.
func spawnPrefetch(logger *log.Logger) {
	exe, err := os.Executable()
	if err != nil {
		logger.Printf("WARNING: could not resolve executable for prefetch: %v", err)
		return
	}

	args := []string{"--prefetch"}
	// Carry over --verbose if present
	for _, arg := range os.Args[1:] {
		if arg == "--verbose" {
			args = append(args, "--verbose")
			break
		}
	}

	cmd := exec.Command(exe, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Start(); err != nil {
		logger.Printf("WARNING: could not spawn prefetch process: %v", err)
		return
	}

	logger.Printf("spawned prefetch process (pid=%d)", cmd.Process.Pid)
	// Detach — don't wait for it
	cmd.Process.Release()
}

// ---------------------------------------------------------------------------
// results.tsv parsing
// ---------------------------------------------------------------------------

type experiment struct {
	Iter        string
	ValBPB      string
	BestValBPB  string
	MemoryGB    string
	Status      string
	Description string
	Timestamp   string
}

func parseResultsTSV(logger *log.Logger) ([]experiment, error) {
	path := os.Getenv("RESULTS_TSV")
	if path == "" {
		path = "results.tsv"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			logger.Printf("no results.tsv found at %s, proceeding without history", path)
			return nil, nil
		}
		return nil, fmt.Errorf("reading results: %w", err)
	}

	reader := csv.NewReader(bytes.NewReader(data))
	reader.Comma = '\t'
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1

	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parsing results TSV: %w", err)
	}

	if len(records) <= 1 {
		logger.Printf("results.tsv has only header, no experiments yet")
		return nil, nil
	}

	var experiments []experiment
	for _, row := range records[1:] {
		if len(row) < 6 {
			continue
		}
		e := experiment{
			Iter:        row[0],
			ValBPB:      row[1],
			BestValBPB:  row[2],
			MemoryGB:    row[3],
			Status:      row[4],
			Description: row[5],
		}
		if len(row) >= 7 {
			e.Timestamp = row[6]
		}
		experiments = append(experiments, e)
	}

	logger.Printf("loaded %d experiments from results.tsv", len(experiments))
	return experiments, nil
}

func formatHistory(experiments []experiment) string {
	if len(experiments) == 0 {
		return "No experiments have been run yet. This is the first iteration."
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "Total experiments so far: %d\n\n", len(experiments))

	var keeps, discards, crashes int
	for _, e := range experiments {
		switch strings.ToLower(strings.TrimSpace(e.Status)) {
		case "keep":
			keeps++
		case "discard":
			discards++
		case "crash":
			crashes++
		}
	}
	fmt.Fprintf(&sb, "Outcomes: %d kept, %d discarded, %d crashed\n", keeps, discards, crashes)

	if len(experiments) > 0 {
		last := experiments[len(experiments)-1]
		fmt.Fprintf(&sb, "Current best val_bpb: %s\n", last.BestValBPB)
	}

	sb.WriteString("\nRecent experiments (last 10):\n")
	start := 0
	if len(experiments) > 10 {
		start = len(experiments) - 10
	}
	for _, e := range experiments[start:] {
		fmt.Fprintf(&sb, "  iter=%s  val_bpb=%s  status=%s  %s\n",
			e.Iter, e.ValBPB, e.Status, e.Description)
	}

	streak := 0
	for i := len(experiments) - 1; i >= 0; i-- {
		if strings.ToLower(strings.TrimSpace(experiments[i].Status)) == "keep" {
			break
		}
		streak++
	}
	if streak > 0 {
		fmt.Fprintf(&sb, "\nWARNING: %d consecutive experiments without improvement.\n", streak)
	}

	return sb.String()
}

// ---------------------------------------------------------------------------
// DeepSeek types & call
// ---------------------------------------------------------------------------

type deepseekRequest struct {
	Model       string            `json:"model"`
	Messages    []deepseekMessage `json:"messages"`
	Stream      bool              `json:"stream"`
	Temperature *float64          `json:"temperature,omitempty"`
}

type deepseekMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type deepseekResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// callDeepSeek calls the DeepSeek API. Pass "" for model to use cfg default.
func callDeepSeek(cfg directorConfig, apiKey, model, systemPrompt, userPrompt string) (string, error) {
	if model == "" {
		model = cfg.DeepSeekModel
	}
	reqBody := deepseekRequest{
		Model:       model,
		Stream:      false,
		Temperature: &cfg.Temperature,
		Messages: []deepseekMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshaling request: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, deepseekAPIURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		appendAPILog(apiLogger, cfg.Name, body, nil, err)
		return "", fmt.Errorf("calling DeepSeek: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		appendAPILog(apiLogger, cfg.Name, body, nil, err)
		return "", fmt.Errorf("reading response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		callErr := fmt.Errorf("DeepSeek returned status %s: %s", resp.Status, string(respBytes))
		appendAPILog(apiLogger, cfg.Name, body, respBytes, callErr)
		return "", callErr
	}

	appendAPILog(apiLogger, cfg.Name, body, respBytes, nil)

	var dsResp deepseekResponse
	if err := json.Unmarshal(respBytes, &dsResp); err != nil {
		return "", fmt.Errorf("decoding response: %w", err)
	}

	if len(dsResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return strings.TrimSpace(dsResp.Choices[0].Message.Content), nil
}

// ---------------------------------------------------------------------------
// HTTP helper
// ---------------------------------------------------------------------------

func doGet(reqURL string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("User-Agent", "autoresearch-director/1.0")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("performing request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("unexpected status: %s", resp.Status)
	}
	return resp, nil
}

// ---------------------------------------------------------------------------
// API call logging (JSONL)
// ---------------------------------------------------------------------------

func logDir() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	exe, err = filepath.EvalSymlinks(exe)
	if err != nil {
		return ""
	}
	repoRoot := filepath.Dir(filepath.Dir(exe))
	return filepath.Join(repoRoot, "director", "logs")
}

type apiLogEntry struct {
	Timestamp  string          `json:"timestamp"`
	Experiment string          `json:"experiment"`
	Request    json.RawMessage `json:"request"`
	Response   json.RawMessage `json:"response,omitempty"`
	Error      string          `json:"error,omitempty"`
}

func appendAPILog(logger *log.Logger, experiment string, reqBody []byte, respBody []byte, callErr error) {
	dir := logDir()
	if dir == "" {
		logger.Println("WARNING: could not resolve log directory, skipping API log")
		return
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		logger.Printf("WARNING: could not create log dir %s: %v", dir, err)
		return
	}

	entry := apiLogEntry{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Experiment: experiment,
		Request:    json.RawMessage(reqBody),
	}
	if callErr != nil {
		entry.Error = callErr.Error()
	}
	if respBody != nil {
		entry.Response = json.RawMessage(respBody)
	}

	line, err := json.Marshal(entry)
	if err != nil {
		logger.Printf("WARNING: could not marshal log entry: %v", err)
		return
	}

	logFile := filepath.Join(dir, "api_calls.jsonl")
	f, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		logger.Printf("WARNING: could not open log file %s: %v", logFile, err)
		return
	}
	defer f.Close()

	f.Write(line)
	f.Write([]byte("\n"))
	logger.Printf("logged API call to %s", logFile)
}

// ---------------------------------------------------------------------------
// .env loading
// ---------------------------------------------------------------------------

func loadAPIKey(logger *log.Logger) string {
	if data, err := os.ReadFile(".env"); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			key, val, ok := strings.Cut(line, "=")
			if !ok {
				continue
			}
			key = strings.TrimSpace(key)
			val = strings.Trim(strings.TrimSpace(val), `"'`)
			if key == "DEEPSEEK_API_KEY" && val != "" {
				logger.Println("loaded DEEPSEEK_API_KEY from .env")
				return val
			}
		}
	}

	if v := os.Getenv("DEEPSEEK_API_KEY"); v != "" {
		logger.Println("using DEEPSEEK_API_KEY from environment")
		return v
	}

	logger.Fatal("DEEPSEEK_API_KEY not found in .env or environment")
	return ""
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	verbose := false
	prefetchMode := false
	for _, arg := range os.Args[1:] {
		switch arg {
		case "--verbose":
			verbose = true
		case "--prefetch":
			prefetchMode = true
		}
	}

	logOutput := io.Discard
	if verbose {
		logOutput = os.Stderr
	}
	logger := log.New(logOutput, "[director] ", log.LstdFlags)
	apiLogger = logger

	cfg := loadConfig(logger)
	apiKey := loadAPIKey(logger)

	// --prefetch mode: fetch + summarize paper, save to cache, exit.
	if prefetchMode {
		runPrefetch(logger, cfg, apiKey)
		return
	}

	// 1. Parse experiment history
	experiments, err := parseResultsTSV(logger)
	if err != nil {
		logger.Printf("WARNING: could not parse results: %v", err)
	}
	history := formatHistory(experiments)

	// 2. Try to load prefetched paper from cache
	var paperTitle, paperAbstract, paperSummary string
	if cache := loadPrefetchCache(logger); cache != nil {
		paperTitle = cache.PaperTitle
		paperAbstract = cache.PaperAbstract
		paperSummary = cache.PaperSummary
		logger.Println("using prefetched paper data")
	} else {
		// No cache — fetch synchronously
		logger.Println("no prefetch cache, fetching paper synchronously")
		var arxivEntry *arxivEntry
		paperTitle, paperAbstract, arxivEntry, err = fetchRandomArxiv(logger, cfg)
		if err != nil {
			logger.Printf("WARNING: arxiv fetch failed: %v, proceeding without paper", err)
			paperTitle = "(no paper fetched)"
			paperAbstract = "No external paper available this round. Generate a directive purely from your knowledge of efficient transformer training techniques."
		}

		// Optionally fetch & summarize the full paper
		if enableFullPaperSummary && arxivEntry != nil {
			paperID := extractArxivID(arxivEntry)
			if paperID != "" {
				fullText, fetchErr := fetchFullPaperText(logger, paperID)
				if fetchErr != nil {
					logger.Printf("WARNING: full paper fetch failed: %v, no paper summary", fetchErr)
				} else {
					pSum, sumErr := summarizePaper(logger, cfg, apiKey, fullText)
					if sumErr != nil {
						logger.Printf("WARNING: paper summarization failed: %v, no paper summary", sumErr)
					} else {
						paperSummary = pSum
						logger.Printf("paper summary generated")
					}
				}
			} else {
				logger.Printf("WARNING: could not extract arxiv ID, no paper summary")
			}
		}
	}

	// 3. Summarize train.py
	trainCode, err := os.ReadFile("train.py")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: could not read train.py: %v\n", err)
		os.Exit(1)
	}

	logger.Println("summarizing train.py...")
	summary, err := callDeepSeek(cfg, apiKey, "deepseek-chat",
		"Summarize this neural network training code in exactly two paragraphs. First paragraph: model architecture (layer types, dimensions, attention mechanism, normalization, activations, positional encoding, parameter count, etc). Second paragraph: optimizer setup, learning rates, batch sizes, scheduling, training loop mechanics, and other stuff. Use exact numbers and names from the code. No opinions, just facts.",
		string(trainCode),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: summarization call failed: %v\n", err)
		os.Exit(1)
	}
	logger.Println("summarization complete")

	// 4. Build the user prompt (template substitution)
	userPrompt := cfg.UserPrompt
	userPrompt = strings.ReplaceAll(userPrompt, "{{history}}", history)
	userPrompt = strings.ReplaceAll(userPrompt, "{{paper_title}}", paperTitle)
	userPrompt = strings.ReplaceAll(userPrompt, "{{paper_abstract}}", paperAbstract)
	if paperSummary != "" {
		userPrompt = strings.ReplaceAll(userPrompt, "{{paper_summary}}", paperSummary)
	} else {
		// Remove the summary section header and placeholder when no summary is available
		userPrompt = strings.ReplaceAll(userPrompt, "\n\n### Detailed Summary\n\n{{paper_summary}}", "")
	}
	userPrompt = strings.ReplaceAll(userPrompt, "{{code_summary}}", summary)

	// 5. Call DeepSeek for directive
	logger.Println("generating directive...")
	result, err := callDeepSeek(cfg, apiKey, "", cfg.SystemPrompt, userPrompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: directive call failed: %v\n", err)
		os.Exit(1)
	}

	logger.Println("directive generated")

	// 6. Output
	fmt.Println(result)

	// 7. Spawn prefetch for next iteration
	spawnPrefetch(logger)
}
