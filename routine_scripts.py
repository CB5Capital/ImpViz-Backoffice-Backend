from regime_model import main as regime_model_script
from feature_updater import main as feature_updater_script

def main(job_type, specific_models):
    specific_models = [int(model) for model in specific_models]

    if job_type == "All":
        feature_updater_script()
        regime_model_script(specific_models)
            
    elif job_type == "FeatureUpdater":
        feature_updater_script()
            
    elif job_type == "RegimeModel":
        regime_model_script(specific_models)