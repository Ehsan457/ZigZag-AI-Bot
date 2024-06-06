import cv2
import numpy as np
import datetime
import os
import shutil
from playwright.async_api import async_playwright
from time import sleep
import asyncio


async def capture_screen(page):
    screenshot = await page.screenshot()
    screenshot = np.frombuffer(screenshot, dtype=np.uint8)
    screenshot = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    return screenshot_gray


async def press_space(page):
    await page.keyboard.press("Space")


def save_image(image, iteration, description):
    os.makedirs("images", exist_ok=True)
    filename = f"images/iteration_{iteration}_{description}.png"
    cv2.imwrite(filename, image)


def detect_ball(screenshot):
    blurred = cv2.GaussianBlur(screenshot, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=10,
        param1=50,
        param2=20,
        minRadius=5,
        maxRadius=15,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            return (x, y, r)
    return None


def create_region(ball_pos, region_range, points_distance, direction):
    slope = 0.5  # adjust the slope to make the region narrower
    if direction == "right":
        new_point = (
            ball_pos[0] + region_range,
            ball_pos[1] - int(region_range * slope),
        )
        side_p1 = [
            ball_pos[0] - points_distance,
            ball_pos[1] - int(points_distance * slope),
        ]
        side_p2 = [
            ball_pos[0] + points_distance,
            ball_pos[1] + int(points_distance * slope),
        ]
        side_p3 = [
            new_point[0] + points_distance,
            new_point[1] + int(points_distance * slope),
        ]
        side_p4 = [
            new_point[0] - points_distance,
            new_point[1] - int(points_distance * slope),
        ]
    elif direction == "left":
        new_point = (
            ball_pos[0] - region_range,
            ball_pos[1] - int(region_range * slope),
        )
        side_p1 = [
            ball_pos[0] + points_distance,
            ball_pos[1] - int(points_distance * slope),
        ]
        side_p2 = [
            ball_pos[0] - points_distance,
            ball_pos[1] + int(points_distance * slope),
        ]
        side_p3 = [
            new_point[0] - points_distance,
            new_point[1] + int(points_distance * slope),
        ]
        side_p4 = [
            new_point[0] + points_distance,
            new_point[1] - int(points_distance * slope),
        ]
    return np.array([side_p1, side_p2, side_p3, side_p4], np.int32)


async def main():
    gameover_template = cv2.imread(r"assets\ScoreBoard.png", cv2.IMREAD_GRAYSCALE)

    if os.path.exists("./images"):
        shutil.rmtree("./images")

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://plays.org/game/zig-zag/")
        await asyncio.sleep(5)
        await press_space(page)  # Skip the initial menu
        await asyncio.sleep(3)

        iteration = 0
        exit_flag = False
        direction = "left"  # Initial assumption
        tmp_mtch_threshold = 0.8  # Threshold for gameover template matching

        canny_t1 = 0
        canny_t2 = 116

        region_range = 22  # Define region range
        points_distance = 2
        radius_to_avoid = 16

        while not exit_flag:
            if iteration == 0:
                await press_space(page)
            big_screen = await capture_screen(page)
            res = cv2.matchTemplate(big_screen, gameover_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= tmp_mtch_threshold)

            if len(loc[0]) == 0:
                print(
                    f"{datetime.datetime.now().timestamp()}\titeration({iteration}) Game is still running."
                )

                small_screen = big_screen[200:600, 400:1000]
                canny_screen = cv2.Canny(small_screen, canny_t1, canny_t2)

                ball_pos = detect_ball(canny_screen)
                print("ball_pos\t", ball_pos)

                if ball_pos:  # If ball detected
                    ball_mask = np.zeros_like(canny_screen)
                    cv2.circle(
                        ball_mask,
                        (ball_pos[0], ball_pos[1]),
                        radius_to_avoid,
                        color=(255, 255, 255),
                        thickness=-1,
                    )

                    region_coords = create_region(
                        ball_pos, region_range, points_distance, direction
                    )
                    region_mask = np.zeros_like(canny_screen)
                    cv2.fillPoly(region_mask, [region_coords], color=(255, 255, 255))
                    vision_mask = region_mask - ball_mask
                    save_image(vision_mask, iteration, "vision_mask")
                    obstacle = cv2.bitwise_and(canny_screen, vision_mask)
                    save_image(obstacle, iteration, "obstacle")
                    save_image(canny_screen, iteration, "canny")

                    # Apply binary threshold to ensure actual_vision is binary
                    _, obstacle_binary = cv2.threshold(
                        obstacle, 1, 255, cv2.THRESH_BINARY
                    )

                    # If the vision of the ball is detecting anything press space
                    if np.any(obstacle_binary == 255):
                        await press_space(page)
                        # print("* space pressed")
                        direction = "left" if direction == "right" else "right"
                iteration += 1

            else:  # if similarity between screen and gameover template found
                print(f"{datetime.datetime.now().timestamp()} Game over!")
                exit_flag = True

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
