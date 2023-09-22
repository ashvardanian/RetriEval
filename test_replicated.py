import numpy as np
from usearch.index import Indexes
from usearch.io import load_matrix
from utils.metrics import recall
from time import perf_counter
from tqdm import tqdm
from multiprocessing import cpu_count

import plotly.graph_objects as go
import numpy as np


INDEX_PATHS = (
    '/mnt/disk1/USearch-HNSW-f32.usearch',
    '/mnt/disk3/USearch-HNSW-f32.usearch',
    '/mnt/disk4/USearch-HNSW-f32.usearch',
    '/mnt/disk5/USearch-HNSW-f32.usearch',
    '/mnt/disk6/USearch-HNSW-f32.usearch',
    '/mnt/disk7/USearch-HNSW-f32.usearch',
    '/mnt/disk8/USearch-HNSW-f32.usearch',
    '/mnt/disk9/USearch-HNSW-f32.usearch',
    '/mnt/disk10/USearch-HNSW-f32.usearch',
    '/mnt/disk11/USearch-HNSW-f32.usearch',
    '/mnt/disk12/USearch-HNSW-f32.usearch',
    '/mnt/disk13/USearch-HNSW-f32.usearch'
)

INDEX_NAME = 'USearch(HNSW,f32)'
DATA_PATH = '/mnt/disk0/datasets/deep/base.1B.fbin'
DATA_SIZE = 1000_000_000


QUERY_SIZES = (10_000, 50_000, 100_000, 500_000, 1000_000, 5000_000, 10_000_000)


def measure(seed=42, k=10):

    '''indices = []
    for index_path in tqdm(INDEX_PATHS, desc='Index loading'):
        indices.append(Index.restore(index_path, view=True))'''
    
    print('Index loading')
    index = Indexes(
        paths=INDEX_PATHS,
        view=True,
        threads=cpu_count()
    )
    print('Loaded!')
    
    # index.search(np.random.random((5000_000, 96)), 10)

    np.random.seed(seed)
    search_speed = []
    recalls = []

    for size in tqdm(QUERY_SIZES):
        start_idx = np.random.randint(0, DATA_SIZE - size)
        groundtruth = np.arange(start_idx, start_idx + size)
        query = load_matrix(DATA_PATH, start_idx, size, view=True)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)

        start_time = perf_counter()
        nn_ids = index.search(query, k).keys
        spent_time = perf_counter() - start_time
        search_speed.append(size / spent_time)

        recall_at_k = float(recall(nn_ids, groundtruth, ats=(k,))[0])
        recalls.append(recall_at_k)

    return search_speed, recalls


def draw_plot(
    x,
    y,
    method,
    x_title,
    y_title,
    title
):
    graphs = [go.Scatter(x=x, y=y, name=method, line=dict(width=3))]

    fig = go.Figure(
        data=graphs,
        layout={
            'xaxis': {'title': x_title},
            'yaxis': {'title': y_title},
            'title': {'text': title},
            'width': 1200,
            'legend': {
                'xanchor': 'center',
                'x': 0.5,
                'y': 1.1,
                'orientation': 'h',
            },
        },
    )
    fig.update_layout(title_x=0.5)

    return fig


if __name__ == '__main__':
    search_speed, recalls = measure()
    np.savez(f'stats/{INDEX_NAME}_replicated.npz', search_speed=search_speed, recalls=recalls)

    fig = draw_plot(
        QUERY_SIZES,
        search_speed,
        INDEX_NAME,
        'num of queries',
        'vecs/s',
        'DEEP 1B x 12: Search Speed'
    )
    fig.write_image('plots/search_speed_replicated.png', scale=2)

    fig = draw_plot(
        QUERY_SIZES,
        recalls,
        INDEX_NAME,
        'num of queries',
        'recall',
        'DEEP 1B x 12: Recall@10'
    )
    fig.write_image('plots/recall_replicated.png', scale=2)