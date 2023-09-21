TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/faiss_usearch.json
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/scann.json
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/qdrant_weaviate.json
TF_CPP_MIN_LOG_LEVEL=3 python run.py configs/exact.json
python utils/draw_plots.py
echo "Done! All plots are in 'plots' directory"