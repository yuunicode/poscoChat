document.addEventListener("DOMContentLoaded", function () {
    const sendBtn = document.getElementById("sendBtn");
    const userInput = document.getElementById("userInput");
    const chatWindow = document.getElementById("chatWindow");

    const settingsBtn = document.getElementById("settingsBtn");
    const settingsDialog = document.getElementById("settingsDialog");
    const closeSettings = document.getElementById("closeSettings");

    const viewLogsBtn = document.getElementById("viewLogsBtn");
    const logsDialog = document.getElementById("logsDialog");
    const logsContent = document.getElementById("logsContent");
    const closeLogs = document.getElementById("closeLogs");

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    settingsBtn.addEventListener("click", () => settingsDialog.classList.remove("hidden"));
    closeSettings.addEventListener("click", () => settingsDialog.classList.add("hidden"));

    viewLogsBtn.addEventListener("click", async () => {
        const res = await fetch("/logs");
        const data = await res.json();
        logsContent.innerHTML = JSON.stringify(data.logs, null, 2);
        logsDialog.classList.remove("hidden");
    });

    closeLogs.addEventListener("click", () => logsDialog.classList.add("hidden"));

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        addMessage(message, "user");
        userInput.value = "";
        sendBtn.disabled = true;

        const colbert = document.getElementById("colbertRerank").checked;
        const crossEnc = document.getElementById("crossEncoder").checked;
        const promptType = document.getElementById("promptType").value;

        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: message,
                colbert_rerank: colbert,
                cross_encoder: crossEnc,
                prompt_type: promptType
            })
        });

        const data = await res.json();
        addMessage(marked.parse(data.answer), "bot");
        sendBtn.disabled = false;
    }

    function addMessage(text, sender) {
        const div = document.createElement("div");
        div.classList.add(sender === "user" ? "user-msg" : "bot-msg");
        div.innerHTML = text;
        chatWindow.appendChild(div);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});
