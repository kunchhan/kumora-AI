import sys
import subprocess


def main():
    print("🌸 Kumora Chat Interface Launcher 🌸")
    print("=" * 40)
    print("1. Terminal Interface")
    print("2. Web Interface (Streamlit)")
    print("3. Exit")
    
    choice = input("\nSelect interface (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting Terminal Interface...")
        subprocess.run([sys.executable, "kumora_chat_terminal.py"])
    elif choice == "2":
        print("\nStarting Web Interface...")
        print("The web interface will open in your browser.")
        subprocess.run([sys.executable, "app.py"])
    elif choice == "3":
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice. Please try again.")
        main()


if __name__ == "__main__":
    main()