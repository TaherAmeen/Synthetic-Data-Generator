import json
import os
import time

from langdetect import detect, DetectorFactory
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

# For consistent language detection
DetectorFactory.seed = 0

# =========================
# USER CONFIG
# =========================
MAX_REVIEWS = 50
OUTPUT_PATH = r"data\real_reviews.json"
BASE_URL = "https://www.capterra.com/p/129281/Easygenerator/reviews/?page="

# =========================
# EXTRACT REVIEWS FROM PAGE
# =========================
def extract_reviews_from_page(driver):
    wait = WebDriverWait(driver, 15)
    
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//h3")))
    except TimeoutException:
        return []

    review_cards = driver.find_elements(By.XPATH, "//h3/ancestor::div[contains(@class,'p-6')]")
    reviews = []

    for card in review_cards:
        # -------------------------
        # TITLE
        # -------------------------
        try:
            title = card.find_element(By.XPATH, ".//h3").text.strip()
        except NoSuchElementException:
            title = ""

        # -------------------------
        # REVIEWER INFO
        # -------------------------
        try:
            info_block = card.find_element(By.XPATH, ".//div[contains(@class,'typo-10')]")
            info_lines = info_block.text.split("\n")
            reviewer_role = info_lines[1].strip() if len(info_lines) > 1 else ""
        except NoSuchElementException:
            reviewer_role = ""

        # -------------------------
        # RATING
        # -------------------------
        try:
            rating = int(float(card.find_element(By.XPATH, ".//div[@data-testid='rating']//span[last()]").text))
        except:
            rating = None

        # -------------------------
        # COMMENT
        # -------------------------
        try:
            comment = card.find_element(By.XPATH, ".//div[contains(@class,'!mt-4')]//p").text.strip()
        except:
            comment = ""

        # -------------------------
        # EXPAND HIDDEN PROS/CONS IF NEEDED
        # -------------------------
        try:
            continue_button = card.find_element(By.XPATH, ".//button[contains(@data-testid,'continue-reading-button')]")
            if continue_button.is_displayed():
                driver.execute_script("arguments[0].click();", continue_button)
                time.sleep(0.3)  # small wait for animation
        except NoSuchElementException:
            pass
        except ElementClickInterceptedException:
            pass

        # -------------------------
        # PROS
        # -------------------------
        try:
            pros = card.find_element(
                By.XPATH,
                ".//span[normalize-space()='Pros']/parent::span/following-sibling::p"
            ).text.strip()
        except:
            pros = ""

        # -------------------------
        # CONS
        # -------------------------
        try:
            cons = card.find_element(
                By.XPATH,
                ".//span[normalize-space()='Cons']/parent::span/following-sibling::p"
            ).text.strip()
        except:
            cons = ""

        # -------------------------
        # LANGUAGE CHECK
        # -------------------------
        combined_text = (pros + " " + cons).strip()
        if combined_text:
            try:
                if detect(combined_text) != 'en':
                    print(f"Skipping non-English review: {title[:50]}...")
                    continue
            except:
                pass

        reviews.append({
            "title": title,
            "reviewer_role": reviewer_role,
            "comment": comment,
            "pros": pros,
            "cons": cons,
            "rating": rating
        })

    return reviews


# =========================
# MAIN SCRAPER
# =========================
def scrape_reviews():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Non-headless Chrome
    driver = webdriver.Chrome()
    all_reviews = []
    page = 1

    try:
        while len(all_reviews) < MAX_REVIEWS:
            print(f"Scraping page {page}")
            driver.get(BASE_URL + str(page))
            time.sleep(2)  # wait for page to load

            page_reviews = extract_reviews_from_page(driver)
            if not page_reviews:
                break

            for review in page_reviews:
                all_reviews.append(review)
                if len(all_reviews) >= MAX_REVIEWS:
                    break

            page += 1

    finally:
        driver.quit()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_reviews)} reviews to {OUTPUT_PATH}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    scrape_reviews()
