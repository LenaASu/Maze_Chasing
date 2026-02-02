import subprocess
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoAlertPresentException, TimeoutException

def run_automation(num_games = 100):
    print("Starting automation script...")
    ai_process = subprocess.Popen(['python', 'navi.py'])

    web_server = subprocess.Popen(['python', '-m', 'http.server', '8000'])
    time.sleep(2)  # Wait for server to start

    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print("Launching web application...")
        driver.get('http://localhost:8000/index.html')

        for i in range(num_games):
            print(f"Starting game {i + 1} of {num_games}...")
            
            try:
                alert = driver.switch_to.alert
                print(f"Cleaning residual alert: {alert.text}")
                alert.accept()
            except NoAlertPresentException:
                pass

            wait = WebDriverWait(driver, 10)
            mcts_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-algorithm="mcts"]')))
            mcts_btn.click()
            print(f"Game {i+1}: MCTS Applied.")
            time.sleep(1)

            # actions = webdriver.ActionChains(driver)
            # actions.send_keys('j')
            # actions.perform()

            # Check if game has ended    
            while True:
                try:
                    # time.sleep(2)  # Wait before checking status
                    alert = driver.switch_to.alert
                    print(f"Game {i + 1} ended with alert: {alert.text}")
                    alert.accept()
                    
                    # Give some time before starting next game
                    time.sleep(3)
                    break
                except NoAlertPresentException:
                    # If no alert, continue checking after a short delay
                    time.sleep(2)  
                    continue
                except Exception as e:
                    print(f"Error during game {i + 1}: {e}")
                    break

        time.sleep(2)  # Wait for the page to load 

        mcts_button = driver.find_element(By.ID, 'mcts')
        mcts_button.click()
        print("MCTS Selected.") 

        start_time = time.time()
        timeout = 300  # 5 minutes

        while time.time() - start_time < timeout:
            # try:
            #     status_element = driver.find_element(By.ID, 'status')
            #     status_text = status_element.text
            #     print(f"Current Status: {status_text}")

            #     if "Game Over" in status_text or "Victory" in status_text:
            #         print("Game ended.")
            #         break

            # except Exception as e:
            #     print(f"Error checking game status: {e}")

            time.sleep(5)  # Check every 5 seconds  
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Closing browser and terminating processes...")
        driver.quit()
        ai_process.terminate()
        web_server.terminate()
        print("Automation script finished.")

if __name__ == "__main__":
    run_automation()