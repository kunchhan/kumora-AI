// // static/main.js

// document.addEventListener("DOMContentLoaded", () => {
//     // --- DOM Elements ---
//     const chatWindow = document.getElementById("chat-window");
//     const chatForm = document.getElementById("chat-form");
//     const messageInput = document.getElementById("message-input");
//     const typingIndicator = document.getElementById("typing-indicator");

//     // --- State Management ---
//     let userId = localStorage.getItem("kumoraUserId");
//     let sessionId = localStorage.getItem("kumoraSessionId");
//     let conversationHistory = [];

//     // Generate IDs if they don't exist
//     if (!userId) {
//         userId = `web_user_${crypto.randomUUID()}`;
//         localStorage.setItem("kumoraUserId", userId);
//     }
//     if (!sessionId) {
//         sessionId = `web_session_${crypto.randomUUID()}`;
//         localStorage.setItem("kumoraSessionId", sessionId);
//     }

//     // --- Functions ---
//     const addMessageToWindow = (text, sender) => {
//         const messageElement = document.createElement("div");
//         messageElement.classList.add("message", `${sender}-message`);
//         messageElement.textContent = text;
//         chatWindow.appendChild(messageElement);
//         // Scroll to the latest message
//         chatWindow.scrollTop = chatWindow.scrollHeight;
//     };

//     const handleFormSubmit = async (event) => {
//         event.preventDefault();
//         const message = messageInput.value.trim();
//         if (!message) return;

//         addMessageToWindow(message, "user");
//         conversationHistory.push({ user: message });
//         messageInput.value = "";
//         typingIndicator.style.display = "flex";

//         try {
//             const response = await fetch("/chat", {
//                 method: "POST",
//                 headers: { "Content-Type": "application/json" },
//                 body: JSON.stringify({
//                     message: message,
//                     userId: userId,
//                     sessionId: sessionId,
//                     history: conversationHistory.slice(-4) // Send last 2 turns
//                 }),
//             });

//             if (!response.ok) {
//                 const errorData = await response.json();
//                 throw new Error(errorData.error || "An unknown error occurred.");
//             }

//             const data = await response.json();
//             const kumoraResponse = data.response;

//             addMessageToWindow(kumoraResponse, "kumora");
//             conversationHistory.push({ kumora: kumoraResponse });

//         } catch (error) {
//             console.error("Error sending message:", error);
//             addMessageToWindow("I'm having a little trouble connecting right now. Please try again in a moment.", "kumora-error");
//         } finally {
//             typingIndicator.style.display = "none";
//         }
//     };

//     // --- Event Listeners ---
//     chatForm.addEventListener("submit", handleFormSubmit);

//     // Initial welcome message
//     addMessageToWindow("Hello! I'm Kumora. I'm here to listen and support you. How are you feeling today?", "kumora");
// });

// static/main.js

// document.addEventListener("DOMContentLoaded", () => {
//     // --- DOM Elements ---
//     const greetingView = document.getElementById("greeting-view");
//     const chatWindowContainer = document.getElementById("chat-window-container");
//     const chatWindow = document.getElementById("chat-window");
//     const chatForm = document.getElementById("chat-form");
//     const messageInput = document.getElementById("message-input");
//     const typingIndicator = document.getElementById("typing-indicator");
//     const newChatBtn = document.getElementById("new-chat-btn");
//     const historyList = document.getElementById("history-list");
//     const promptStarterBtns = document.querySelectorAll(".prompt-starter-btn");

//     // --- State Management ---
//     let userId = localStorage.getItem("kumoraUserId") || `web_user_${crypto.randomUUID()}`;
//     let sessionId;
//     let conversationHistory = [];
//     let isChatActive = false;

//     localStorage.setItem("kumoraUserId", userId);

//     const startNewChat = () => {
//         sessionId = `web_session_${crypto.randomUUID()}`;
//         conversationHistory = [];
//         isChatActive = false;
        
//         chatWindow.innerHTML = ''; // Clear chat window
//         greetingView.style.display = 'flex';
//         chatWindowContainer.style.display = 'none';
//         messageInput.value = '';
//         messageInput.focus();
//     };

//     const activateChatView = () => {
//         if (!isChatActive) {
//             greetingView.style.display = 'none';
//             chatWindowContainer.style.display = 'flex';
//             isChatActive = true;
//         }
//     };

//     const addMessageToWindow = (text, sender, isTyping = false) => {
//         const messageElement = document.createElement("div");
//         messageElement.classList.add("message", `${sender}-message`);
        
//         // Simple markdown for bold and italics
//         text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
//         text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
//         messageElement.innerHTML = text;

//         chatWindow.appendChild(messageElement);
//         chatWindowContainer.scrollTop = chatWindowContainer.scrollHeight;
//     };

//     const sendMessage = async (message) => {
//         if (!message) return;

//         activateChatView();
//         addMessageToWindow(message, "user");
//         conversationHistory.push({ user: message });
//         messageInput.value = "";
//         typingIndicator.style.display = "flex";

//         try {
//             const response = await fetch("/chat", {
//                 method: "POST",
//                 headers: { "Content-Type": "application/json" },
//                 body: JSON.stringify({
//                     message: message,
//                     userId: userId,
//                     sessionId: sessionId,
//                     history: conversationHistory.slice(-4)
//                 }),
//             });

//             if (!response.ok) throw new Error("Network response was not ok.");

//             const data = await response.json();
//             addMessageToWindow(data.response, "kumora");
//             conversationHistory.push({ kumora: data.response });

//         } catch (error) {
//             console.error("Error sending message:", error);
//             addMessageToWindow("I'm having a little trouble connecting right now. Please try again.", "kumora");
//         } finally {
//             typingIndicator.style.display = "none";
//         }
//     };

//     // Auto-resize textarea
//     messageInput.addEventListener('input', () => {
//         messageInput.style.height = 'auto';
//         messageInput.style.height = (messageInput.scrollHeight) + 'px';
//     });

//     // --- Event Listeners ---
//     chatForm.addEventListener("submit", (e) => {
//         e.preventDefault();
//         sendMessage(messageInput.value.trim());
//     });
    
//     newChatBtn.addEventListener("click", startNewChat);

//     promptStarterBtns.forEach(btn => {
//         btn.addEventListener("click", () => {
//             const prompt = btn.getAttribute("data-prompt");
//             sendMessage(prompt);
//         });
//     });

//     // Initialize the first chat session
//     startNewChat();
// });

// static/main.js

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const sidebar = document.getElementById("sidebar");
    const menuToggle = document.getElementById("menu-toggle");
    const brandHomeBtn = document.getElementById("brand-home-btn");
    const newChatIconBtn = document.getElementById("new-chat-icon-btn");
    const newChatTextBtn = document.getElementById("new-chat-text-btn");
    const greetingView = document.getElementById("greeting-view");
    const chatWindowContainer = document.getElementById("chat-window-container");
    const chatWindow = document.getElementById("chat-window");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const typingIndicator = document.getElementById("typing-indicator");
    const newChatBtn = document.getElementById("new-chat-btn");
    const historyList = document.getElementById("history-list");
    const promptStarterBtns = document.querySelectorAll(".prompt-starter-btn");

    // --- State Management ---
    let userId = localStorage.getItem("kumoraUserId") || `web_user_${crypto.randomUUID()}`;
    let sessionId;
    let conversationHistory = [];
    let isChatActive = false;

    localStorage.setItem("kumoraUserId", userId);

    const startNewChat = () => {
        sessionId = `web_session_${crypto.randomUUID()}`;
        conversationHistory = [];
        isChatActive = false;
        
        chatWindow.innerHTML = '';
        greetingView.style.display = 'flex';
        chatWindowContainer.style.display = 'none';
        messageInput.value = '';
        messageInput.focus();
        
        // On mobile, close sidebar after starting a new chat
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('is-open');
        }
    };

    const activateChatView = () => {
        if (!isChatActive) {
            greetingView.style.display = 'none';
            chatWindowContainer.style.display = 'block'; // Changed from flex
            isChatActive = true;
        }
    };

    const addMessageToWindow = (text, sender) => {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>');
        messageElement.innerHTML = text;

        chatWindow.appendChild(messageElement);
        chatWindowContainer.scrollTop = chatWindowContainer.scrollHeight;
    };

    const sendMessage = async (message) => {
        if (!message) return;

        activateChatView();
        addMessageToWindow(message, "user");
        conversationHistory.push({ user: message });
        messageInput.value = "";
        messageInput.style.height = 'auto'; // Reset textarea height
        typingIndicator.style.display = "flex";

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: message,
                    userId: userId,
                    sessionId: sessionId,
                    history: conversationHistory.slice(-4)
                }),
            });

            if (!response.ok) throw new Error("Network response was not ok.");

            const data = await response.json();
            addMessageToWindow(data.response, "kumora");
            conversationHistory.push({ kumora: data.response });

        } catch (error) {
            console.error("Error sending message:", error);
            addMessageToWindow("I'm having a little trouble connecting right now. Please try again.", "kumora");
        } finally {
            typingIndicator.style.display = "none";
        }
    };

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = (messageInput.scrollHeight) + 'px';
    });

    // --- Event Listeners ---
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        sendMessage(messageInput.value.trim());
    });
    
    brandHomeBtn.addEventListener("click", startNewChat);
    newChatIconBtn.addEventListener("click", startNewChat);
    newChatTextBtn.addEventListener("click", startNewChat);

    promptStarterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const prompt = btn.getAttribute("data-prompt");
            sendMessage(prompt);
        });
    });

    // Sidebar toggle for mobile
    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('is-open');
    });

    // Initialize the first chat session
    startNewChat();
});