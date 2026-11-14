"""
Comprehensive End-to-End Playwright Browser Verification Script
Tests the running Streamlit application at http://localhost:8501.
"""

import sys
import time
from playwright.sync_api import sync_playwright

def run_browser_verification():
    print("Starting Playwright browser verification...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Test 1: Desktop Viewport (1280x800)
        print("\n--- Test 1: Desktop Viewport (1280x800) ---")
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto("http://localhost:8501", timeout=30000)
        
        # Wait for Streamlit to render the full overview page
        page.wait_for_selector("button:has-text('Begin Health Assessment')", timeout=15000)
        time.sleep(1)
        text_content = page.inner_text("body")

        # Verify Overview Page text
        assert "Multi-Disease Classification System" in text_content, "Missing main title"
        assert "Supported Conditions" in text_content, "Missing Supported Conditions section"
        assert "Diabetes Mellitus" in text_content
        assert "Heart Disease (CHD / Myocardial Infarction)" in text_content
        assert "Stroke" in text_content
        assert "Asthma" in text_content
        assert "Chronic Obstructive Pulmonary Disease (COPD)" in text_content
        assert "Chronic Kidney Disease" in text_content
        assert "Arthritis" in text_content
        print("PASS: Overview page title, description, and 7 supported conditions verified.")

        # Absence checks
        for forbidden in ["CodeAlpha", "Task 4", "Internship", "Remediated", "Important Educational Classification & Clinical Boundary Notice"]:
            assert forbidden.lower() not in text_content.lower(), f"Found forbidden term: {forbidden}"
        print("PASS: Verified absence of CodeAlpha, Task 4, Internship, Remediated, and old disclaimer.")

        # Verify removed pages are absent
        assert "Model Performance" not in text_content
        assert "About the Dataset" not in text_content
        print("PASS: Verified absence of 'Model Performance' and 'About the Dataset' pages.")

        # Test Navigation: Click 'Begin Health Assessment' button
        print("\n--- Test 2: Navigation to Assessment via button ---")
        btn = page.locator("button:has-text('Begin Health Assessment')")
        assert btn.is_visible(), "Begin Health Assessment button not visible"
        btn.click()
        
        # Wait for Assessment page heading and bottom submit button
        page.wait_for_selector("button:has-text('Run Assessment')", timeout=15000)
        time.sleep(1)
        text_content = page.inner_text("body")
        assert "Section A — Demographic Information" in text_content
        assert "Section B — General Health" in text_content
        assert "Section C — Lifestyle" in text_content
        assert "Section D — Healthcare Access" in text_content
        print("PASS: Successfully navigated to Assessment page; all 4 questionnaire sections visible.")

        # Test Form Submission
        print("\n--- Test 3: Form Submission and Prediction Execution ---")
        submit_btn = page.locator("button:has-text('Run Assessment')")
        assert submit_btn.is_visible(), "Run Assessment button not visible"
        submit_btn.click()
        
        # Wait for Assessment Results
        page.wait_for_selector("text=Assessment Results", timeout=15000)
        time.sleep(2)
        text_content = page.inner_text("body")
        print("DEBUG TEXT CONTENT AFTER RUN ASSESSMENT:\n", text_content[-1500:])
        
        assert "SimpleImputer" not in text_content, "Error: SimpleImputer error appeared on page!"
        assert "_fill_dtype" not in text_content, "Error: _fill_dtype error appeared on page!"
        assert "Error executing prediction" not in text_content, "Prediction execution error found!"

        # Check that table contains all 7 conditions
        for cond in ["Diabetes Mellitus", "Heart Disease (CHD / MI)", "Stroke", "Asthma", "COPD / Chronic Bronchitis", "Chronic Kidney Disease", "Arthritis"]:
            assert cond in text_content, f"Condition {cond} not found in results table"
        print("PASS: All 7 conditions evaluated without error; results table rendered cleanly.")

        # Check footer
        assert "Educational classification project using self-reported survey data. Not intended for medical use." in text_content
        print("PASS: Footer disclaimer verified.")

        # Test 4: Preset loading (Senior profile)
        print("\n--- Test 4: Preset Loading (Senior Profile) ---")
        expander = page.locator("text=Load sample profile (optional)")
        expander.click()
        time.sleep(1)
        senior_btn = page.locator("button:has-text('Senior profile')")
        senior_btn.click()
        time.sleep(2)
        page.wait_for_selector("button:has-text('Run Assessment')", timeout=15000)
        
        # Submit senior profile
        page.locator("button:has-text('Run Assessment')").click()
        time.sleep(2)
        page.wait_for_selector("text=Assessment Results", timeout=15000)
        senior_text = page.inner_text("body")
        assert "Positive" in senior_text, "Senior profile should produce positive classifications"
        print("PASS: Senior profile submitted and produced positive classifications.")

        # Test 5: Mobile / Narrow Viewport (375x667)
        print("\n--- Test 5: Mobile Viewport (375x667) ---")
        page_mobile = browser.new_page(viewport={"width": 375, "height": 667})
        page_mobile.goto("http://localhost:8501", timeout=30000)
        page_mobile.wait_for_selector("button:has-text('Begin Health Assessment')", timeout=15000)
        mobile_text = page_mobile.inner_text("body")
        assert "Multi-Disease Classification System" in mobile_text
        print("PASS: Mobile layout loads cleanly without layout breakages.")

        browser.close()
        print("\n=======================================================")
        print("ALL BROWSER E2E INTERACTION TESTS PASSED SUCCESSFULLY!")
        print("=======================================================")

if __name__ == "__main__":
    run_browser_verification()
