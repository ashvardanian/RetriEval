rm -rf  ~/.cache/weaviate-embedded

TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/usearch_faiss.json
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/scann.json
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/qdrant_weaviate.json

rm -rf stats/*.chunk_*.npz
python utils/draw_plots.py

echo "Done! All plots are in plots directory"