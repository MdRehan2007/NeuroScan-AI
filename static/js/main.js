document.addEventListener("DOMContentLoaded", () => {
  const uploadDropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzoneDefault = document.getElementById("dropzone-default");
  const dropzonePreview = document.getElementById("dropzone-preview");
  const previewImg = document.getElementById("preview-img");
  const fileNameEl = document.getElementById("file-name");

  const predictBtn = document.getElementById("predict-btn");
  const resetBtn = document.getElementById("reset-btn");
  const errorBox = document.getElementById("error-box");
  const spinner = document.getElementById("spinner");

  // Prediction Results Card
  const resultsCard = document.getElementById("results-card");
  const resultClass = document.getElementById("result-class");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceFill = document.getElementById("confidence-fill");
  const resultMriImg = document.getElementById("result-mri-img");

  const probGliomaVal = document.getElementById("prob-glioma-val");
  const probGliomaFill = document.getElementById("prob-glioma-fill");
  const probMeningiomaVal = document.getElementById("prob-meningioma-val");
  const probMeningiomaFill = document.getElementById("prob-meningioma-fill");
  const probPituitaryVal = document.getElementById("prob-pituitary-val");
  const probPituitaryFill = document.getElementById("prob-pituitary-fill");
  const probNotumorVal = document.getElementById("prob-notumor-val");
  const probNotumorFill = document.getElementById("prob-notumor-fill");

  // Explainability Card Section
  const explainabilityCard = document.getElementById("explainability-card");

  const exGradcamImg = document.getElementById("ex-gradcam-img");
  const exGradcamErr = document.getElementById("ex-gradcam-err");

  const exGradcamppImg = document.getElementById("ex-gradcampp-img");
  const exGradcamppErr = document.getElementById("ex-gradcampp-err");

  const exScorecamImg = document.getElementById("ex-scorecam-img");
  const exScorecamErr = document.getElementById("ex-scorecam-err");

  const exLayercamImg = document.getElementById("ex-layercam-img");
  const exLayercamErr = document.getElementById("ex-layercam-err");

  const exLimeImg = document.getElementById("ex-lime-img");
  const exLimeErr = document.getElementById("ex-lime-err");

  const exShapImg = document.getElementById("ex-shap-img");
  const exShapErr = document.getElementById("ex-shap-err");

  let selectedFile = null;

  function resetResult() {
    resultsCard.style.display = "none";
    explainabilityCard.style.display = "none";

    errorBox.style.display = "none";
    errorBox.textContent = "";

    confidenceValue.textContent = "0%";
    confidenceFill.style.width = "0%";

    probGliomaVal.textContent = "0.00%";
    probGliomaFill.style.width = "0%";
    probMeningiomaVal.textContent = "0.00%";
    probMeningiomaFill.style.width = "0%";
    probPituitaryVal.textContent = "0.00%";
    probPituitaryFill.style.width = "0%";
    probNotumorVal.textContent = "0.00%";
    probNotumorFill.style.width = "0%";

    resetExCard(exGradcamImg, exGradcamErr);
    resetExCard(exGradcamppImg, exGradcamppErr);
    resetExCard(exScorecamImg, exScorecamErr);
    resetExCard(exLayercamImg, exLayercamErr);
    resetExCard(exLimeImg, exLimeErr);
    resetExCard(exShapImg, exShapErr);
  }

  function resetExCard(imgEl, errEl) {
    if (imgEl) {
      imgEl.src = "";
      imgEl.style.display = "none";
    }
    if (errEl) {
      errEl.style.display = "none";
      errEl.textContent = "Unavailable";
    }
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.style.display = "block";
  }

  function handleFile(file) {
    if (!file) return;

    const allowed = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowed.includes(file.type)) {
      showError("Please select a JPG, JPEG, or PNG image.");
      return;
    }

    selectedFile = file;
    resetResult();

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropzoneDefault.style.display = "none";
      dropzonePreview.style.display = "flex";
    };
    reader.readAsDataURL(file);

    fileNameEl.textContent = file.name;
    predictBtn.disabled = false;
  }

  uploadDropzone.addEventListener("click", (e) => {
    if (e.target !== fileInput) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    uploadDropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadDropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    uploadDropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadDropzone.classList.remove("dragover");
    });
  });

  uploadDropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      handleFile(file);
    }
  });

  resetBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dropzonePreview.style.display = "none";
    dropzoneDefault.style.display = "flex";
    fileNameEl.textContent = "";
    predictBtn.disabled = true;
    resetResult();
  });

  function updateExCard(imgEl, errEl, url, err) {
    if (url && imgEl) {
      imgEl.src = url;
      imgEl.style.display = "block";
      if (errEl) errEl.style.display = "none";
    } else {
      if (imgEl) imgEl.style.display = "none";
      if (errEl) {
        errEl.textContent = err || "Unavailable";
        errEl.style.display = "block";
      }
    }
  }

  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("Please choose an MRI image first.");
      return;
    }

    resetResult();
    spinner.style.display = "block";
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        showError(data.error || "Something went wrong while predicting.");
        return;
      }

      // 1. Prediction details
      resultClass.textContent = data.prediction;
      confidenceValue.textContent = (data.confidence || 0).toFixed(2) + "%";
      if (resultMriImg) resultMriImg.src = data.image_url;

      // 2. Class Probabilities
      const probs = data.probabilities || {};
      probGliomaVal.textContent = (probs.glioma || 0).toFixed(2) + "%";
      probMeningiomaVal.textContent = (probs.meningioma || 0).toFixed(2) + "%";
      probPituitaryVal.textContent = (probs.pituitary || 0).toFixed(2) + "%";
      probNotumorVal.textContent = (probs.notumor || 0).toFixed(2) + "%";

      resultsCard.style.display = "block";

      requestAnimationFrame(() => {
        confidenceFill.style.width = (data.confidence || 0) + "%";
        probGliomaFill.style.width = (probs.glioma || 0) + "%";
        probMeningiomaFill.style.width = (probs.meningioma || 0) + "%";
        probPituitaryFill.style.width = (probs.pituitary || 0) + "%";
        probNotumorFill.style.width = (probs.notumor || 0) + "%";
      });

      // 3. 6 Explainability Cards
      const ex = data.explanations || {};
      const errors = data.explanation_errors || {};

      updateExCard(exGradcamImg, exGradcamErr, data.gradcam_url || ex.gradcam, data.gradcam_error || errors.gradcam);
      updateExCard(exGradcamppImg, exGradcamppErr, data.gradcampp_url || ex.gradcampp, errors.gradcampp);
      updateExCard(exScorecamImg, exScorecamErr, data.scorecam_url || ex.scorecam, errors.scorecam);
      updateExCard(exLayercamImg, exLayercamErr, data.layercam_url || ex.layercam, errors.layercam);
      updateExCard(exLimeImg, exLimeErr, data.lime_url || ex.lime, data.lime_error || errors.lime);
      updateExCard(exShapImg, exShapErr, data.shap_url || ex.shap, errors.shap);

      explainabilityCard.style.display = "block";

    } catch (err) {
      showError("Could not reach the server. Please check that it is running.");
    } finally {
      spinner.style.display = "none";
      predictBtn.disabled = false;
    }
  });
});
