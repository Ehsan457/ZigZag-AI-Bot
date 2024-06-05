import cv2
import numpy as np
import datetime
import os
import shutil
from playwright.sync_api import sync_playwright
from time import sleep


def capture_screen(page):

    screenshot = page.screenshot()
    screenshot = np.frombuffer(screenshot, dtype=np.uint8)
    screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    return screenshot_gray


def press_space(page):
    page.keyboard.press("Space")


def save_image(image, iteration, description):
    os.makedirs("images", exist_ok=True)
    filename = f"images/iteration_{iteration}_{description}.png"
    cv2.imwrite(filename, image)


def main():
    gameover_template = cv2.imread(r"assets\ScoreBoard.png", cv2.IMREAD_GRAYSCALE)

    if os.path.exists("./images"):
        shutil.rmtree("./images")

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        page.goto("https://plays.org/game/zig-zag/")
        sleep(3)
        press_space(page)  # Skip the initial menu
        sleep(1)
        press_space(page)  # Start the game

        ####### Variables Here

        iteration = 0
        exit_flag = False
        direction = "right"  # Initial assumption
        tmp_mtch_threshold = 0.8  # Threshold for gameover template matching

        canny_t1 = 50
        canny_t2 = 100
        #######

        while not exit_flag:
            # The idea is to only use big screen for template matching
            big_screen = capture_screen(page)
            # crop big screen into smaller screen to avoid redundancy
            small_screen = big_screen[200:600, 400:1000]
            save_image(small_screen, iteration, "screen")
            res = cv2.matchTemplate(big_screen, gameover_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= tmp_mtch_threshold)

            if len(loc[0]) == 0:
                print(
                    f"{datetime.datetime.now().timestamp()}\titeration({iteration}) Game is still running."
                )
                canny_screen = cv2.Canny(small_screen, canny_t1, canny_t2)
                save_image(canny_screen, iteration, "canny")

                iteration += 1

            else:
                print(f"{datetime.datetime.now().timestamp()} Game over!")
                exit_flag = True

        browser.close()


if __name__ == "__main__":
    main()
