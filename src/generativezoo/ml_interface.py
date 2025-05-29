import requests
import plotly.graph_objects as go
from simian.gui import Form, component, utils
import logging
import base64

# Add form and current model as global variables at the top
form = None
current_model = "Vanilla VAE"
model_trained = False  # Initialize model_trained as False

# Add logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from simian.local import Uiformio

    Uiformio("ml_interface", window_title="GenerativeZoo Models Interface")

# API Endpoints
BASE_API_URL = ""
VAE_TRAIN_API_URL = f"{BASE_API_URL}/vanillavae_train"
VAE_SAMPLE_API_URL = f"{BASE_API_URL}/vanillavae_sample"
DDPM_TRAIN_API_URL = f"{BASE_API_URL}/DDPMs_train"
DDPM_SAMPLE_API_URL = f"{BASE_API_URL}/DDPMs_sample"
AR_TRAIN_API_URL = f"{BASE_API_URL}/Autoregressive_train"
AR_SAMPLE_API_URL = f"{BASE_API_URL}/Autoregressive_sample"
GAN_TRAIN_API_URL = f"{BASE_API_URL}/GAN_train"
GAN_SAMPLE_API_URL = f"{BASE_API_URL}/GAN_sample"
FLOW_TRAIN_API_URL = f"{BASE_API_URL}/Flow_train"
FLOW_SAMPLE_API_URL = f"{BASE_API_URL}/Flow_sample"
SGM_TRAIN_API_URL = f"{BASE_API_URL}/SGM_train"
SGM_SAMPLE_API_URL = f"{BASE_API_URL}/SGM_sample"
DOWNLOAD_MODEL_URL = f"{BASE_API_URL}/ModelDownload"

# Model parameter configurations
MODEL_PARAMS = {
    "Vanilla SGM": ["n_epochs", "batch_size", "lr"],
    "NCSNv2": ["n_epochs", "batch_size", "lr"],
    "Vanilla VAE": ["n_epochs", "batch_size", "lr"],
    "Hierarchical VAE": ["n_epochs", "batch_size", "lr"],
    "Conditional VAE": ["n_epochs", "batch_size", "lr"],
    "Vanilla DDPM": ["n_epochs", "batch_size", "lr"],
    "Conditional DDPM": ["n_epochs", "batch_size", "lr"],
    "DAE": ["n_epochs", "batch_size", "lr"],
    "PixelCNN": ["n_epochs", "batch_size", "lr"],
    "VQ-GAN + Transformer": ["n_epochs", "batch_size", "lr"],
    "VQ-VAE + Transformer": ["n_epochs", "batch_size", "lr"],
    "Adversarial VAE": ["n_epochs", "batch_size", "lr"],
    "Vanilla GAN": ["n_epochs", "batch_size", "lr"],
    "Conditional GAN": ["n_epochs", "batch_size", "lr"],
    "CycleGAN": ["n_epochs", "batch_size", "lr"],
    "Prescribed GAN": ["n_epochs", "batch_size", "lr"],
    "Wasserstein GAN with Gradient Penalty": ["n_epochs", "batch_size", "lr"],
    "Vanilla Flow": ["n_epochs", "batch_size", "lr"],
    "RealNVP": ["n_epochs", "batch_size", "lr"],
    "Glow": ["n_epochs", "batch_size", "lr"],
    "Flow++": ["n_epochs", "batch_size", "lr"],
    "Flow Matching": ["n_epochs", "batch_size", "lr"],
    "Conditional Flow Matching": ["n_epochs", "batch_size", "lr"],
    "Rectified Flows": ["n_epochs", "batch_size", "lr"]
}

def create_parameter_input(param_name: str, param_config: dict, parent_container) -> component.Component:
    """Create an input component based on parameter configuration"""
    if param_config["type"] == "number":
        input_comp = component.Number(param_name, parent_container)
        input_comp.defaultValue = param_config["default"]
        input_comp.step = param_config.get("step", 1)
    elif param_config["type"] == "select":
        input_comp = component.Select(param_name, parent_container)
        input_comp.data = {"values": param_config["options"]}
        input_comp.defaultValue = param_config["default"]
        input_comp.valueProperty = "value"
    else:
        input_comp = component.TextInput(param_name, parent_container)
        input_comp.defaultValue = param_config["default"]

    input_comp.label = param_config["label"]
    input_comp.tooltip = param_config.get("tooltip", f"Set the {param_config['label']}")
    return input_comp


def gui_init(meta_data: dict) -> dict:
    global current_model
    form = Form()

    # Create a container for better layout
    container = component.Well("main_container", form)
    container.addCustomClass("container mt-4")

    # Add a header with description
    header = component.HtmlElement("header", container)
    header.tag = "div"
    header.content = """
        <div class="text-center mb-4">
            <h2>GenerativeZoo Models Interface</h2>
            <p class="text-muted">Train and sample from various generative models on different datasets</p>
        </div>
    """

    # Create main training panel
    training_panel = component.Panel("training_panel", container)
    training_panel.title = "Model Training Configuration"
    training_panel.collapsible = True
    training_panel.theme = "primary"

    # Model Selection Components
    model_select = component.Select("model_name", training_panel)
    model_select.label = "Select Model Architecture"
    model_select.data = {"values": [{"label": name, "value": name} for name in MODEL_PARAMS.keys()]}
    model_select.valueProperty = "value"
    model_select.defaultValue = current_model
    model_select.tooltip = "Choose the generative model architecture to train"
    model_select.setRequired()
    model_select.properties = {"triggerHappy": "model_changed"}

    dataset_select = component.Select("dataset", training_panel)
    dataset_select.label = "Training Dataset"
    dataset_select.data = {"values": [
        {"label": name.upper(), "value": name} for name in [
            "mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist",
            "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn",
            "tinyimagenet", "imagenet"
        ]
    ]}
    dataset_select.valueProperty = "value"
    dataset_select.defaultValue = "mnist"
    dataset_select.tooltip = "Select the dataset to train the model on"

    # Common Parameters Panel
    common_panel = component.Panel("common_panel", training_panel)
    common_panel.title = "Training Parameters"
    common_panel.collapsible = True
    common_panel.theme = "secondary"

    epochs_input = component.Number("n_epochs", common_panel)
    epochs_input.label = "Number of Epochs"
    epochs_input.defaultValue = 100
    epochs_input.tooltip = "Set the number of training epochs"
    epochs_input.add_validation(integer=True)

    batch_size_input = component.Number("batch_size", common_panel)
    batch_size_input.label = "Batch Size"
    batch_size_input.defaultValue = 256
    batch_size_input.tooltip = "Set the batch size for training"
    batch_size_input.add_validation(integer=True)

    lr_input = component.Number("lr", common_panel)
    lr_input.label = "Learning Rate"
    lr_input.defaultValue = 0.001
    lr_input.tooltip = "Set the learning rate for training"
    lr_input.add_validation(step=0.0001)

    # Training Button
    train_button = component.Button("train_model", training_panel)
    train_button.label = "Train Model"
    train_button.action = "event"
    train_button.event = "train_model"
    train_button.theme = "primary"
    train_button.leftIcon = "fas fa-cog"
    train_button.tooltip = "Start training the model with the selected configuration"

    # Training Status
    train_status = component.Html("training_status", training_panel)
    train_status.tag = "div"
    train_status.defaultValue = """
        <div class="d-flex align-items-center alert alert-info mt-3">
            <i class="fas fa-info-circle me-2"></i>
            <span>Training Status: Not Started</span>
        </div>
    """

    # Model Download Panel
    download_panel = component.Panel("download_panel", container)
    download_panel.title = "Model Management"
    download_panel.collapsible = True
    download_panel.theme = "info"

    # Download Status
    download_status = component.Html("download_status", download_panel)
    download_status.tag = "div"
    download_status.defaultValue = """
        <div class="d-flex align-items-center alert alert-info mb-3">
            <i class="fas fa-info-circle me-2"></i>
            <span>Select a model and dataset to download</span>
        </div>
    """

    # Download Button
    download_button = component.Button("download_model", download_panel)
    download_button.label = "Download Model"
    download_button.action = "event"
    download_button.event = "download_model"
    download_button.theme = "info"
    download_button.leftIcon = "fas fa-download"
    download_button.tooltip = "Download the trained model weights and configuration"
    download_button.block = True
    download_button.size = "lg"

    # Sampling Panel
    sampling_panel = component.Panel("sampling_panel", container)
    sampling_panel.title = "Model Sampling"
    sampling_panel.collapsible = True
    sampling_panel.theme = "success"

    # Sampling Button
    sample_button = component.Button("sample_model", sampling_panel)
    sample_button.label = "Generate Samples"
    sample_button.action = "event"
    sample_button.event = "sample_model"
    sample_button.theme = "success"
    sample_button.leftIcon = "fas fa-image"
    sample_button.tooltip = "Generate samples from the trained model"
    sample_button.block = True
    sample_button.size = "lg"

    # Sampling Status
    sample_status = component.Html("sample_status", sampling_panel)
    sample_status.tag = "div"
    sample_status.defaultValue = """
        <div class="d-flex align-items-center alert alert-info mt-3">
            <i class="fas fa-info-circle me-2"></i>
            <span>Sampling Status: Not Started</span>
        </div>
    """

    # Sample Image Display
    image_container = component.Well("image_container", sampling_panel)
    image_container.addCustomClass("mt-3")

    sample_image = component.Html("sample_image", image_container)
    sample_image.defaultValue = """
        <div class="text-center p-4 bg-light rounded">
            <i class="fas fa-image fa-3x text-muted mb-3"></i>
            <p class="text-muted">Generated images will appear here</p>
        </div>
    """

    return {
        "form": form,
        "navbar": {
            "title": "Deep Generative Models Interface",
            "color": "primary",
        },
        "showChanged": True
    }


def model_changed(meta_data: dict, payload: dict) -> dict:
    """Handle model selection change event"""
    global current_model, model_trained

    # Get the new model selection and update current model
    new_model, _ = utils.getSubmissionData(payload, "model_name")
    logger.info(f"Model changed from {current_model} to {new_model}")

    if new_model:
        current_model = new_model
        model_trained = False  # Reset model_trained when model changes
        logger.info(f"Current model updated to: {current_model}")
    else:
        logger.error("No model selected in payload")
        return payload

    # Clear any existing training status
    training_status = """
        <div class="d-flex align-items-center alert alert-info">
            <i class="fas fa-info-circle me-2"></i>
            <span>Training Status: Not Started</span>
        </div>
    """
    utils.setSubmissionData(payload, "training_status", training_status)

    # Reset sample status and image
    sample_status = """
        <div class="d-flex align-items-center alert alert-info">
            <i class="fas fa-info-circle me-2"></i>
            <span>Sampling Status: Not Started</span>
        </div>
    """
    utils.setSubmissionData(payload, "sample_status", sample_status)

    sample_image = """
        <div class="text-center p-4 bg-light rounded">
            <i class="fas fa-image fa-3x text-muted mb-3"></i>
            <p class="text-muted">Generated images will appear here</p>
        </div>
    """
    utils.setSubmissionData(payload, "sample_image", sample_image)

    return payload


def gui_event(meta_data: dict, payload: dict) -> dict:
    Form.eventHandler(train_model=train_model)
    Form.eventHandler(sample_model=sample_model)
    Form.eventHandler(model_changed=model_changed)
    Form.eventHandler(download_model=download_model)

    callback = utils.getEventFunction(meta_data, payload)
    return callback(meta_data, payload)


# 📌 Model eğitme fonksiyonu
def train_model(meta_data: dict, payload: dict) -> dict:
    global current_model, model_trained

    # Get the current model from payload
    selected_model, _ = utils.getSubmissionData(payload, "model_name")
    if selected_model:
        if selected_model != current_model:
            logger.warning(f"Model mismatch: selected={selected_model}, current={current_model}")
            current_model = selected_model

    logger.info(f"Training model: {current_model}")

    # Get model configuration
    model_params = MODEL_PARAMS.get(current_model)
    if not model_params:
        error_html = f"""
            <div class="d-flex align-items-center alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                <span>Model configuration not found for {current_model}</span>
            </div>
        """
        utils.setSubmissionData(payload, "training_status", error_html)
        model_trained = False
        logger.error(f"Model configuration not found for {current_model}")
        return payload

    # Initialize data dictionary with common parameters
    dataset_value, _ = utils.getSubmissionData(payload, "dataset")
    if not dataset_value:
        error_html = """
            <div class="d-flex align-items-center alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                <span>Please select a dataset</span>
            </div>
        """
        utils.setSubmissionData(payload, "training_status", error_html)
        model_trained = False
        logger.error("Dataset not selected")
        return payload

    data = {
        "model_name": current_model,
        "dataset": dataset_value
    }

    # Validate and add common parameters
    missing_params = []
    for param in model_params:  # Now model_params is just a list of common parameters
        value, _ = utils.getSubmissionData(payload, param)
        if value is None:
            missing_params.append(param)
            continue

        try:
            if param in ["n_epochs", "batch_size"]:
                data[param] = int(value)
            else:
                data[param] = float(value)
        except (ValueError, TypeError):
            error_html = f"""
                <div class="d-flex align-items-center alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                    <span>Invalid value for {param}</span>
                </div>
            """
            utils.setSubmissionData(payload, "training_status", error_html)
            model_trained = False
            logger.error(f"Invalid value for parameter: {param}")
            return payload

    # Check for missing parameters
    if missing_params:
        error_html = f"""
            <div class="d-flex align-items-center alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                <span>Please fill in the following parameters: {', '.join(missing_params)}</span>
            </div>
        """
        utils.setSubmissionData(payload, "training_status", error_html)
        model_trained = False
        logger.error(f"Missing parameters: {missing_params}")
        return payload

    # Determine API endpoint based on model type
    api_url = None
    if current_model in ["Vanilla VAE", "Hierarchical VAE", "Conditional VAE", "Adversarial VAE"]:
        api_url = VAE_TRAIN_API_URL
    elif current_model in ["Vanilla DDPM", "Conditional DDPM", "DAE"]:
        api_url = DDPM_TRAIN_API_URL
    elif current_model in ["PixelCNN", "VQ-GAN + Transformer", "VQ-VAE + Transformer"]:
        api_url = AR_TRAIN_API_URL
    elif current_model in ["Vanilla GAN", "Conditional GAN", "CycleGAN", "Prescribed GAN", "Wasserstein GAN with Gradient Penalty"]:
        api_url = GAN_TRAIN_API_URL
    elif current_model in ["Vanilla Flow", "RealNVP", "Glow", "Flow++", "Flow Matching", "Conditional Flow Matching", "Rectified Flows"]:
        api_url = FLOW_TRAIN_API_URL
    elif current_model in ["Vanilla SGM", "NCSNv2"]:
        api_url = SGM_TRAIN_API_URL

    logger.info(f"Selected API URL for training: {api_url}")
    logger.info(f"Training data: {data}")

    try:
        res = requests.post(api_url, json=data)
        logger.info(f"Training API response status: {res.status_code}")

        if res.status_code == 200:
            message = res.json().get("message", "Model trained successfully!")
            model_trained = True
            logger.info("Model training successful")
            status_html = """
                <div class="d-flex align-items-center alert alert-success">
                    <i class="fas fa-check-circle me-2 text-success"></i>
                    <span>Training Status: Model Successfully Trained! 🎉</span>
                </div>
            """
            utils.setSubmissionData(payload, "training_status", status_html)
        else:
            message = f"API Error: {res.status_code} - {res.text}"
            model_trained = False
            logger.error(f"Training API error: {message}")
            status_html = f"""
                <div class="d-flex align-items-center alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                    <span>Training Status: Failed - {message}</span>
                </div>
            """
            utils.setSubmissionData(payload, "training_status", status_html)

    except Exception as e:
        error_html = f"""
            <div class="d-flex align-items-center alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                <span>Training Status: Error - {str(e)}</span>
            </div>
        """
        utils.setSubmissionData(payload, "training_status", error_html)
        model_trained = False
        logger.error(f"Exception during training: {str(e)}", exc_info=True)

    return payload


def sample_model(meta_data: dict, payload: dict) -> dict:
    global model_trained

    logger.info(f"Sample model called. Model trained: {model_trained}")

    if not model_trained:
        status_html = """
            <div class="d-flex align-items-center alert alert-danger">
                <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                <span>Please train the model first!</span>
            </div>
        """
        utils.setSubmissionData(payload, "sample_status", status_html)
        return payload

    # Get model name and dataset
    model_name, _ = utils.getSubmissionData(payload, "model_name")
    dataset, _ = utils.getSubmissionData(payload, "dataset")

    logger.info(f"Attempting to sample from model: {model_name} with dataset: {dataset}")

    # Initialize data dictionary
    data = {
        "model_name": model_name,
        "dataset": dataset,
        "num_samples": 16  # Default number of samples
    }

    # Determine API endpoint based on model type
    api_url = None
    if model_name in ["Vanilla VAE", "Hierarchical VAE", "Conditional VAE", "Adversarial VAE"]:
        api_url = VAE_SAMPLE_API_URL
    elif model_name in ["Vanilla DDPM", "Conditional DDPM", "DAE"]:
        api_url = DDPM_SAMPLE_API_URL
    elif model_name in ["PixelCNN", "VQ-GAN + Transformer", "VQ-VAE + Transformer"]:
        api_url = AR_SAMPLE_API_URL
    elif model_name in ["Vanilla GAN", "Conditional GAN", "CycleGAN", "Prescribed GAN", "Wasserstein GAN with Gradient Penalty"]:
        api_url = GAN_SAMPLE_API_URL
    elif model_name in ["Vanilla Flow", "RealNVP", "Glow", "Flow++", "Flow Matching", "Conditional Flow Matching", "Rectified Flows"]:
        api_url = FLOW_SAMPLE_API_URL
    elif model_name in ["Vanilla SGM", "NCSNv2"]:
        api_url = SGM_SAMPLE_API_URL

    logger.info(f"Selected API URL: {api_url}")

    if api_url is None:
        error_html = f"""
            <div class="d-flex align-items-center alert alert-danger">
                <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                <span>No sampling endpoint available for model: {model_name}</span>
            </div>
        """
        utils.setSubmissionData(payload, "sample_status", error_html)
        return payload

    try:
        # Update status to "Generating..."
        generating_html = """
            <div class="d-flex align-items-center alert alert-info">
                <i class="fas fa-spinner fa-spin me-2"></i>
                <span>Generating samples...</span>
            </div>
        """
        utils.setSubmissionData(payload, "sample_status", generating_html)

        logger.info(f"Sending request to {api_url} with data: {data}")
        res = requests.post(api_url, json=data)
        logger.info(f"Received response with status code: {res.status_code}")

        try:
            response_json = res.json()
            logger.info(f"Response JSON: {response_json}")
        except Exception as e:
            logger.error(f"Failed to parse response as JSON: {str(e)}")
            logger.error(f"Response content: {res.content}")
            raise

        if res.status_code == 200 and "base64_image" in response_json:
            base64_image = response_json["base64_image"]

            # Handle if response is a list
            if isinstance(base64_image, list):
                base64_image = base64_image[0]
                logger.info("Received multiple images, using the first one")

            # Remove data URL prefix if present
            if "data:image" in base64_image:
                base64_image = base64_image.split(',')[1]
                logger.info("Removed data URL prefix from base64 image")

            # Create the full data URL
            img_url = f"data:image/png;base64,{base64_image}"

            # Create HTML for displaying the image
            img_html = f"""
                <div style="text-align:center;">
                    <p><b>Generated Sample</b></p>
                    <img src="{img_url}" 
                         style="max-width:100%; border:1px solid #ccc; border-radius:8px;" 
                         alt="Generated Sample" />
                </div>
            """

            utils.setSubmissionData(payload, "sample_image", img_html)

            # Update status to success
            success_html = """
                <div class="d-flex align-items-center alert alert-success">
                    <i class="fas fa-check-circle me-2 text-success"></i>
                    <span>Samples generated successfully!</span>
                </div>
            """
            utils.setSubmissionData(payload, "sample_status", success_html)

        else:
            error_message = response_json.get('message', 'Unknown error')
            logger.error(f"API Error: {error_message}")
            error_html = f"""
                <div class="d-flex align-items-center alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                    <span>API Error: {error_message}</span>
                </div>
            """
            utils.setSubmissionData(payload, "sample_status", error_html)

    except Exception as e:
        logger.error(f"Exception during sampling: {str(e)}", exc_info=True)
        error_html = f"""
            <div class="d-flex align-items-center alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                <span>Error: {str(e)}</span>
            </div>
        """
        utils.setSubmissionData(payload, "sample_status", error_html)

    return payload


def download_model(meta_data: dict, payload: dict) -> dict:
    """Handle model download request"""
    # Get model name
    model_name, _ = utils.getSubmissionData(payload, "model_name")

    logger.info(f"Attempting to download model: {model_name}")

    if not model_name:
        error_html = """
            <div class="d-flex align-items-center alert alert-danger">
                <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                <span>Please select a model to download</span>
            </div>
        """
        utils.setSubmissionData(payload, "download_status", error_html)
        return payload

    # Get dataset name
    dataset, _ = utils.getSubmissionData(payload, "dataset")
    if not dataset:
        error_html = """
            <div class="d-flex align-items-center alert alert-danger">
                <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                <span>Please select a dataset</span>
            </div>
        """
        utils.setSubmissionData(payload, "download_status", error_html)
        return payload

    try:
        # Update status to "Checking..."
        checking_html = """
            <div class="d-flex align-items-center alert alert-info">
                <i class="fas fa-spinner fa-spin me-2"></i>
                <span>Checking model availability...</span>
            </div>
        """
        utils.setSubmissionData(payload, "download_status", checking_html)

        # Test if the model exists and can be downloaded
        test_response = requests.post(DOWNLOAD_MODEL_URL, json={"Model": model_name, "Dataset": dataset}, stream=True)
        logger.info(f"Download test response status: {test_response.status_code}")

        if test_response.status_code == 200:
            # Check if we received a ZIP file
            content_type = test_response.headers.get('content-type', '')
            content_disp = test_response.headers.get('content-disposition', '')
            
            if 'application/zip' in content_type or '.zip' in content_disp:
                # Create download form with success message and download location info
                download_html = f"""
                    <div class="d-flex flex-column alert alert-success">
                        <div class="d-flex align-items-center justify-content-between mb-2">
                            <div>
                                <i class="fas fa-check-circle me-2 text-success"></i>
                                <span>Model verified and ready for download!</span>
                            </div>
                            <form action="{DOWNLOAD_MODEL_URL}" method="post" target="_blank" onsubmit="showDownloadStarted(event)">
                                <input type="hidden" name="Model" value="{model_name}">
                                <input type="hidden" name="Dataset" value="{dataset}">
                                <button type="submit" class="btn btn-sm btn-success">
                                    <i class="fas fa-download me-1"></i>
                                    Download Model
                                </button>
                            </form>
                        </div>
                        <div class="small text-muted">
                            <i class="fas fa-info-circle me-1"></i>
                            The file will be downloaded as <strong>{model_name}_{dataset}.zip</strong> to your browser's default download location
                        </div>
                    </div>
                    <script>
                    function showDownloadStarted(event) {{
                        const statusDiv = document.getElementById('download_status');
                        if (statusDiv) {{
                            statusDiv.innerHTML = `
                                <div class="d-flex flex-column">
                                    <div class="d-flex align-items-center mb-2">
                                        <i class="fas fa-spinner fa-spin me-2"></i>
                                        <span>Download started...</span>
                                    </div>
                                    <div class="small">
                                        <i class="fas fa-folder-open me-1"></i>
                                        File <strong>{model_name}_{dataset}.zip</strong> is being downloaded to your browser's default download folder.<br>
                                        <i class="fas fa-info-circle me-1"></i>
                                        You can find this folder by:
                                        <ul class="mb-0 mt-1">
                                            <li>Chrome: Settings → Downloads</li>
                                            <li>Firefox: Options → General → Downloads</li>
                                            <li>Edge: Settings → Downloads</li>
                                        </ul>
                                    </div>
                                </div>
                            `;
                        }}
                    }}
                    </script>
                """
                utils.setSubmissionData(payload, "download_status", download_html)
            else:
                logger.error(f"Unexpected content type: {content_type}")
                error_html = """
                    <div class="d-flex align-items-center alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                        <span>Error: Server did not return a valid model file</span>
                    </div>
                """
                utils.setSubmissionData(payload, "download_status", error_html)
        else:
            error_message = test_response.json().get('detail', 'Unknown error')
            logger.error(f"Download API error: {error_message}")
            error_html = f"""
                <div class="d-flex align-items-center alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2 text-danger"></i>
                    <span>Download Error: {error_message}</span>
                </div>
            """
            utils.setSubmissionData(payload, "download_status", error_html)

    except Exception as e:
        logger.error(f"Exception during download check: {str(e)}", exc_info=True)
        error_html = f"""
            <div class="d-flex align-items-center alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2 text-warning"></i>
                <span>Error checking model: {str(e)}</span>
            </div>
        """
        utils.setSubmissionData(payload, "download_status", error_html)

    return payload
