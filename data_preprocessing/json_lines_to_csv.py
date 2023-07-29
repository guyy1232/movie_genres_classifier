import json
import pandas as pd

MOVIE_FIELDS = {
    'countries': {},
    'feature_length': '',
    'genres': {},
    'languages': {},
    'movie_box_office_revenue': '',
    'plot_summary': '',
    'release_date': '',
    'title': ''
}


def read_json_lines_file(path):
    """
    read the data from the raw json lines file, add missing fields and remove keys from dict type fields.
    :param path:
    :return:
    """
    with open(path, 'r') as file:
        lines = file.readlines()
    json_lines = [json.loads(line) for line in lines]
    for line in json_lines:
        for field_name, empty_value in MOVIE_FIELDS.items():
            line.setdefault(field_name, empty_value)
            if type(line[field_name]) is dict:
                line[field_name] = list(line[field_name].values())

    return json_lines


def json_lines_to_csv(movies, csv_path):
    data_df = pd.DataFrame(movies)
    data_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    processed_lines = read_json_lines_file(path='../data/train.json')
    json_lines_to_csv(processed_lines, '../data/processed_data.csv')
