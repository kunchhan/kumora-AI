// Simple emoji picker implementation
document.addEventListener('DOMContentLoaded', function() {
    const emojiBtn = document.getElementById('emoji-btn');
    const chatInput = document.getElementById('chat-input');
    const emojiPicker = document.getElementById('emoji-picker');
    
    // Common emojis for Nepal/IT context
    const emojis = [
      '😀', '😂', '😊', '😍', '👍', '❤️', '🙏', '🎉', '🤔', '😎',
      '🇳🇵', '💻', '📱', '🔧', '🛠️', '👨‍💻', '👩‍💻', '🏢', '🌐', '📚',
      '🕉️', '☸️', '🐂', '⛰️', '🌄', '🍛', '🥟', '☕', '📈', '💡'
    ];
    
    // Create emoji buttons
    emojis.forEach(emoji => {
      const span = document.createElement('span');
      span.className = 'emoji';
      span.textContent = emoji;
      span.addEventListener('click', () => {
        const cursorPos = chatInput.selectionStart;
        chatInput.value = chatInput.value.substring(0, cursorPos) + 
                         emoji + 
                         chatInput.value.substring(cursorPos);
        chatInput.focus();
        chatInput.selectionStart = chatInput.selectionEnd = cursorPos + emoji.length;
      });
      emojiPicker.appendChild(span);
    });
    
    // Toggle emoji picker
    emojiBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      emojiPicker.classList.toggle('show');
    });
    
    // Close picker when clicking outside
    document.addEventListener('click', () => {
      emojiPicker.classList.remove('show');
    });
    
    // Prevent closing when clicking inside picker
    emojiPicker.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  });