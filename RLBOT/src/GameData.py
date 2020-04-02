import time
import requests
import os
from datetime import datetime
from pathlib import Path


class BallChasingInteractor():
    SECONDS_IN_MINUTE = 60
    CALLS_PER_MINUTE = 2
    AUTHORIZATION = "LNyeFi2IH57vXF7xq8TBvlWy9GPkA7fi7X28D4OG"
    BASE_URL = "https://ballchasing.com/api"
    def __init__(self):
        self._set_last_call_time()
        self.next_url = ""

    def _set_last_call_time(self):
        self.last_call_time = datetime.now() 

    def _seconds_until_next_call(self):
        current_time = datetime.now()
        return (current_time - self.last_call_time).total_seconds()

    def _should_we_make_call(self):
        time_difference = self._seconds_until_next_call()
        seconds_to_wait = self.SECONDS_IN_MINUTE / self.CALLS_PER_MINUTE
        return time_difference >= seconds_to_wait and time_difference > 0

    def _wait_to_make_call(self):
        if not self._should_we_make_call(): 
            print("sleeping for API limit")
            time.sleep(self._seconds_until_next_call())

    def _get_headers(self):
        return { "Authorization" : self.AUTHORIZATION, "X-Remote-IP": "127.0.0.1" }
    
    def _make_call(self, http_verb, url):
        self._wait_to_make_call()
        result = requests.request(http_verb, url, headers=self._get_headers())
        self._set_last_call_time()
        return result

    def _write_replay_to_file(self, response, file_path):
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)        

    def get_replays(self):
        path = "/replays"
        url = f"{self.BASE_URL}{path}?"
        #TODO: is it supposed to be min_rank or min-rank?, can we add the 1v1 only here ?
        url_params = { "10": "on", "season": 12, "min_rank" : "champion-3", "max-rank": "grand-champion"}
        for key, value in url_params.items():
            url = f"{url}{key}={value}&"
        #if there is a next_url start from the initial get_replays call
        if self.next_url: url = self.next_url
        print("grabbing replays to download... \n\t url: {}".format(url))
        return self._make_call("GET", url).json()

    def download_all(self):
        json_response = self.get_replays()
        replay_list = json_response['list']
        next_url = json_response['next']
        self.next_url = next_url
        count = 0
        while self.next_url or count < 3:
            self.download_replay_group(replay_list)
            count+=1 


    def download_replay_group(self, replay_list):
        for replay in replay_list:
            replay_id = replay['id']
            replay_season = replay['season']
            self.download_replay(replay_id, replay_season)
    
    
    def download_replay(self, id, season):
        path = "/replays"
        url = f"https://ballchasing.com/dl/replay/{id}"
        print("downloading replay id {0}".format(id))
        response = requests.post(url)
        if response.status_code == 200:
            home = str(Path.home())
            directory = "{0}\\rocket-league-replays\\season-{1}".format(home, season)
            file_name = "game-id_{0}.replay".format(id)
            file_path = "{0}\\{1}".format(directory, file_name)
            print("download successful, writing to file {}".format(file_path))
            self._write_replay_to_file(response, file_path)

BallChasingInteractor().download_all()