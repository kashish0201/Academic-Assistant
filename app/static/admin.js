const ADMIN_KEY_STORAGE = "csu_admin_key";

const loginScreen = document.getElementById("login-screen");
const adminApp = document.getElementById("admin-app");
const loginForm = document.getElementById("login-form");
const adminKeyInput = document.getElementById("admin-key");
const loginError = document.getElementById("login-error");
const logoutBtn = document.getElementById("logout-btn");
const fileInput = document.getElementById("file-input");
const uploadZone = document.getElementById("upload-zone");
const ingestBtn = document.getElementById("ingest-btn");
const chunkCountEl = document.getElementById("chunk-count");
const pendingFilesEl = document.getElementById("pending-files");
const toastEl = document.getElementById("toast");

function getAdminKey() {
  return sessionStorage.getItem(ADMIN_KEY_STORAGE);
}

function showToast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.hidden = false;
  toastEl.classList.toggle("error", isError);
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    toastEl.hidden = true;
  }, 3500);
}

async function adminFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Admin-Key", getAdminKey());

  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === "string" ? detail : "Request failed";
    throw new Error(message);
  }

  return data;
}

function showAdmin() {
  loginScreen.hidden = true;
  adminApp.hidden = false;
  refreshChunkCount();
}

function showLogin() {
  loginScreen.hidden = false;
  adminApp.hidden = true;
}

async function verifyKey(key) {
  const response = await fetch("/admin/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: key }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "Invalid admin key");
  }
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.hidden = true;

  const key = adminKeyInput.value.trim();
  try {
    await verifyKey(key);
    sessionStorage.setItem(ADMIN_KEY_STORAGE, key);
    showAdmin();
  } catch (error) {
    loginError.textContent = error.message;
    loginError.hidden = false;
  }
});

logoutBtn.addEventListener("click", () => {
  sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  adminKeyInput.value = "";
  showLogin();
});

async function refreshChunkCount() {
  try {
    const data = await adminFetch("/ingest/status");
    chunkCountEl.textContent = String(data.indexed_chunks_in_db ?? "—");

    const pending = data.pending_files || [];
    pendingFilesEl.innerHTML = pending.length
      ? pending.map((name) => `<li>${name}</li>`).join("")
      : '<li class="hint">None — all uploads are indexed</li>';
  } catch {
    chunkCountEl.textContent = "—";
    pendingFilesEl.innerHTML = '<li class="hint">Unable to load status</li>';
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const data = await adminFetch("/upload", { method: "POST", body: formData });
  chunkCountEl.textContent = String(data.indexed_chunks_in_db ?? "—");
  showToast(data.message || "File indexed successfully");
}

async function ingestAll() {
  const data = await adminFetch("/ingest", { method: "POST" });
  chunkCountEl.textContent = String(data.indexed_chunks_in_db ?? "—");
  await refreshChunkCount();
  showToast(data.message || "Ingestion complete");
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  try {
    await uploadFile(file);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    fileInput.value = "";
  }
});

uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", async (event) => {
  event.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  try {
    await uploadFile(file);
  } catch (error) {
    showToast(error.message, true);
  }
});

ingestBtn.addEventListener("click", async () => {
  ingestBtn.disabled = true;
  try {
    await ingestAll();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    ingestBtn.disabled = false;
  }
});

if (getAdminKey()) {
  verifyKey(getAdminKey())
    .then(showAdmin)
    .catch(() => {
      sessionStorage.removeItem(ADMIN_KEY_STORAGE);
      showLogin();
    });
} else {
  showLogin();
}
