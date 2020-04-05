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
    #TODO: is it supposed to be min_rank or min-rank?, can we add the 1v1 only here ?
    REPLAY_PARAMS = { "season": 12, "min_rank" : "champion-3", "max-rank": "grand-champion", "playlist": "ranked-solo-standard" }

    def __init__(self):
        """
        This class is used to download the replay files from ballchasing.com
        Ballchasing has an API rate limit of two calls per minute
        This class is poorly implemented in that it acts as both an API call manager - (deciding when to make the calls) 
        While it also makes the calls and downloads the files.
        It further manages two API's - rocket-league's 
        The files are saved to ~/rocket-league-replays/season-<season>/replay_id-<replay_id>.replay
            :param self:
        """
        self._set_last_call_time()
        self.next_url = ""

    def _set_last_call_time(self):
        """
        private method used for the api rate limit managememnt
        sets the a class variable to keep track of when the last API call was made
            :param self: 
            @return [Void/None]
        """   
        self.last_call_time = datetime.now() 

    def _seconds_since_last_call(self):
        """
        private method used for the api rate limit management
        does the calculation for how much time has elapsed since the last api call
            :param self: 
            @return - [Int] - the amount of seconds that have elapsed since the last api call
        """   
        current_time = datetime.now()
        return  (current_time - self.last_call_time).total_seconds()

    def _should_we_make_call(self):
        """
        private method used for the api rate limit management
        helps us to evenly space out the API calls - in particular, 
        tells us if we will hit the rate limit based on whether 
        enough time has elapsed since the last call that was made 
            :param self:
            @return [Boolean] - whether or not enough time has elapsed since last call 
        """
        seconds_since_last_call = self._seconds_since_last_call()
        seconds_between_calls = self.SECONDS_IN_MINUTE / self.CALLS_PER_MINUTE
        return seconds_since_last_call >= seconds_between_calls and seconds_since_last_call > 0

    def _seconds_to_sleep(self):
        """
        private method used for the api rate limit management
        tells us the amount of time we will need to wait
            :param self:
            @return [Int] - the amount of second we will need to sleep until we can 
                            make the next call without hitting the rate limit
        """   
        seconds_between_calls = self.SECONDS_IN_MINUTE / self.CALLS_PER_MINUTE
        return (seconds_between_calls - self._seconds_since_last_call())

    def _wait_to_make_call(self):
        """
        private method used for the api rate limit management
        if we need to wait due to the api rate limit, then this method
        essentially blocks the thread and stops any code from further executing until
        enough time has elapsed since the last call
            :param self: 
            @return [Void/None]
        """
        if not self._should_we_make_call(): 
            print("sleeping for API limit - sleeping for {} seconds".format(self._seconds_to_sleep()))
            seconds_to_sleep = self._seconds_to_sleep()
            time.sleep(seconds_to_sleep)

    def _get_headers(self):
        """
        private method used for accessing API
        stores and returns the headers - currently static    
            :param self: 
            @return [Dictionary] - the header names and their values
        """ 
        return { "Authorization" : self.AUTHORIZATION, "X-Remote-IP": "127.0.0.1" }
    
    def _make_call(self, http_verb, url):
        """
        private method used for accessing API
        uses the requests library in order to make an HTTP call to the provided URL
            :param self: 
            :param http_verb:   [String] - the HTTP method GET, POST, PUT, DELETE
            :param url:         [String] - the URL that we want to make the request to
            @return - [Requests::Response] - the response object that represents the results of the HTTP call
        """   
        self._wait_to_make_call()
        result = requests.request(http_verb, url, headers=self._get_headers())
        self._set_last_call_time()
        return result

    def _write_replay_to_file(self, response, file_path):
        """
        private method used to save the downloaded file
            :param self: 
            :param response: [Requests::Response] - the response object representing the results of the API call
            :param file_path: [String] - the absolute path where we want to save the file contents to
            @return [Void/None]
        """   
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)        

    def get_replay_list(self):
        """
        Get's a list of replays from the API - the list is paginated
        so if we have already made the call for the first pagination, 
        there will be a url available for the next page of Replays
            :param self:
            @return - [Dictionary] - A Dictionary of the data returned from the API
                                    will contain a list of the replays within this dictionary
                                    for shape of dict, see - https://ballchasing.com/doc/api#replays-replays 
        """   
        path = "/replays"
        url = f"{self.BASE_URL}{path}?"
        for key, value in self.REPLAY_PARAMS.items():
            url = f"{url}{key}={value}&"
        # if there is a next_url then we don't want to make the initial call
        # we want to make the call for the next batch of replay ids to downloadS
        if self.next_url: url = self.next_url
        print("grabbing replays to download... \n\t url: {}".format(url))
        return self._make_call("GET", url).json()

    def download_all(self):
        """
        Honestly this should probably be the only public method
        This kicks off the process of downloading all the replays available 
            :param self: 
            @return [Void/None]
        """   
        json_response = self.get_replay_list()
        replay_list = json_response['list']
        next_url = json_response['next']
        self.next_url = next_url
        count = 0
        while self.next_url or count < 3:
            self.download_replay_group(replay_list)
            count+=1 


    def download_replay_group(self, replay_list):
        """
        After we have retrieved a list of replays from the API
        We need to download each replay from the list of replays
            :param self: 
            :param replay_list: [Array] - an array of dictionaries. each dict will contain data on one replay
            @return [Void/None]
        """
        for replay in replay_list:
            replay_id = replay['id']
            replay_season = replay['season']
            self.download_replay(replay_id, replay_season)
            
    def download_replay(self, id, season):
        """
        Uses a different API to download an individual replay
            :param self: 
            :param id:      [String] - the id of the replay
            :param season:  [Int] - the season this replay belongs to -used for storage organization
            @return [Void/None] 
        """
        path = "/replays"
        url = f"https://ballchasing.com/dl/replay/{id}"
        print("downloading replay id {0}".format(id))
        response = self._make_call("POST", url)
        if response.status_code == 200:
            home = str(Path.home())
            directory = "{0}\\rocket-league-replays\\season-{1}".format(home, season)
            file_name = "game-id_{0}.replay".format(id)
            file_path = "{0}\\{1}".format(directory, file_name)
            print("download successful, writing to file {}".format(file_path))
            self._write_replay_to_file(response, file_path)

BallChasingInteractor().download_all()