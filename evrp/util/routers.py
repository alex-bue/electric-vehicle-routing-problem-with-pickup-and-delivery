import requests

class MapboxValhalla:
    def __init__(self, access_token):
        self.base_url = "https://api.mapbox.com"
        self.access_token = access_token

    def directions(self, profile, coordinates, exclude=None):
        """
        Request directions using the Mapbox API and return a Direction instance containing the raw response.

        :param profile: Routing profile such as 'mapbox/driving', 'mapbox/walking', etc.
        :param coordinates: A list of (longitude, latitude) pairs.
        :param exclude: String or list of strings to specify road types or features to avoid.
        :return: An instance of Direction containing the API response.
        """
        coords_str = ';'.join([f"{lon},{lat}" for lon, lat in coordinates])
        endpoint = f"{self.base_url}/directions/v5/{profile}/{coords_str}"
        params = {
            'access_token': self.access_token,
            'steps': 'true',
            'geometries': 'geojson',
            'overview': 'full'
        }

        # Handle the 'exclude' parameter if provided
        if exclude:
            if isinstance(exclude, list):
                exclude = ','.join(exclude)  # Convert list to comma-separated string
            params['exclude'] = exclude

        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return Direction(response.json())
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")

    def matrix(self, profile, coordinates, sources=None, destinations=None, annotations="duration,distance"):
        """
        Request a distance or duration matrix using the Mapbox API and return a Matrix instance containing the raw response.
        :param profile: Routing profile such as 'mapbox/driving', 'mapbox/walking', etc.
        :param coordinates: A list of (longitude, latitude) pairs.
        :param sources: Use the coordinates at a given index as sources. Possible values are: a semicolon-separated list of 0-based indices, or all (default). The option all allows using all coordinates as sources.
        :param destinations: Use the coordinates at a given index as destinations. Possible values are: a semicolon-separated list of 0-based indices, or all (default). The option all allows using all coordinates as destinations.
        :param annotations: Used to specify the resulting matrices. Possible values are: duration (default), distance, or both values separated by a comma.
        :return: A Matrix with durations, distances and raw json response.
        """
        coords_str = ';'.join([f"{lon},{lat}" for lon, lat in coordinates])
        endpoint = f"{self.base_url}/directions-matrix/v1/{profile}/{coords_str}"
        params = {
            'access_token': self.access_token,
            'annotations': annotations
        }

        if sources is not None:
            if isinstance(sources, list):
                sources = ';'.join(map(str, sources))
            params['sources'] = sources

        if destinations is not None:
            if isinstance(destinations, list):
                destinations = ';'.join(map(str, destinations))
            params['destinations'] = destinations

        response = requests.get(endpoint, params=params)
        if response.status_code == 200:
            return self._parse_matrix_response(response.json())
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")

    def _parse_matrix_response(self, response_json):
        durations = response_json.get('durations')
        distances = response_json.get('distances')
        return Matrix(durations=durations, distances=distances, raw=response_json)


class Direction:
    def __init__(self, json_data):
        self._raw = json_data

        try:
            if 'routes' not in json_data or not json_data['routes']:
                raise ValueError("No routes found in response.")

            route = json_data['routes'][0]
            self._distance = route['distance']  # in meters
            self._duration = route['duration']  # in seconds

            # Extract coordinates directly from the GeoJSON geometry object
            if 'geometry' in route and 'coordinates' in route['geometry']:
                self._geometry = route['geometry']['coordinates']
            else:
                raise ValueError("No valid geometry found in response.")
        except KeyError as e:
            raise ValueError(f"Missing expected data in the API response: {e}")
        except IndexError as e:
            raise ValueError("No route data found in the API response.") from e

    @property
    def raw(self):
        """Returns the raw JSON response from the API."""
        return self._raw

    @property
    def geometry(self):
        return self._geometry

    @property
    def distance(self):
        return self._distance

    @property
    def duration(self):
        return self._duration

    def __str__(self):
        return f"Direction(distance={self.distance}m, duration={self.duration}s, geometry={self.geometry})"


class Matrix:
    def __init__(self, durations=None, distances=None, raw=None):
        self._durations = durations
        self._distances = distances
        self._raw = raw

    @property
    def durations(self):
        return self._durations

    @property
    def distances(self):
        return self._distances

    @property
    def raw(self):
        return self._raw

    def __str__(self):
        return f"Matrix(durations={self.durations}, distances={self.distances})"

