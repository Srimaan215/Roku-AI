"""
Roku GUI - Daily Driver Interface
A clean, desktop interface for daily use with multi-LoRA stress testing.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QScrollArea,
    QFrame, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.personalized_roku import PersonalizedRoku


class RokuWorker(QThread):
    """Background worker for LLM inference"""
    finished = pyqtSignal(str, dict)  # response, metadata
    error = pyqtSignal(str)
    
    def __init__(self, roku: PersonalizedRoku, message: str, history: List[Dict]):
        super().__init__()
        self.roku = roku
        self.message = message
        self.history = history
        self.start_time = None
    
    def run(self):
        try:
            self.start_time = datetime.now()
            response = self.roku.chat(
                message=self.message,
                history=self.history,
                max_tokens=256
            )
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            metadata = {
                "latency": elapsed,
                "adapters": self.roku.llm.active_adapters,
                "adapter_scales": self.roku.llm.adapter_info,
                "timestamp": datetime.now().isoformat()
            }
            
            self.finished.emit(response, metadata)
        except Exception as e:
            self.error.emit(str(e))


class RokuGUI(QMainWindow):
    """Main GUI for Roku AI"""
    
    def __init__(self):
        super().__init__()
        self.roku = None
        self.history = []
        self.conversation_log = []
        self.worker = None
        
        self.init_ui()
        self.init_roku()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Roku AI - Daily Driver")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Chat interface
        chat_panel = self.create_chat_panel()
        
        # Right panel: Adapter monitoring
        monitor_panel = self.create_monitor_panel()
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(chat_panel)
        splitter.addWidget(monitor_panel)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Initializing Roku...")
    
    def create_chat_panel(self) -> QWidget:
        """Create the main chat interface"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("💬 Roku AI Assistant")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setStyleSheet("padding: 10px; background-color: #2c3e50; color: white; border-radius: 5px;")
        layout.addWidget(header)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Monaco", 11))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setFont(QFont("Arial", 12))
        self.input_field.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 2px solid #3e3e3e;
                border-radius: 5px;
                background-color: #2b2b2b;
                color: #d4d4d4;
            }
            QLineEdit:focus {
                border: 2px solid #007acc;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self.clear_history)
        
        self.save_button = QPushButton("Save Conversation")
        self.save_button.clicked.connect(self.save_conversation)
        
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.save_button)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        return panel
    
    def create_monitor_panel(self) -> QWidget:
        """Create the monitoring panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("📊 System Monitor")
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header.setStyleSheet("padding: 8px; background-color: #34495e; color: white; border-radius: 5px;")
        layout.addWidget(header)
        
        # Active adapters section
        adapters_group = QGroupBox("Active LoRA Adapters")
        adapters_layout = QVBoxLayout()
        
        self.adapter_labels = {}
        for adapter_name in ["personality", "personal", "health", "work"]:
            label = QLabel(f"❌ {adapter_name}: Inactive")
            label.setFont(QFont("Monaco", 10))
            self.adapter_labels[adapter_name] = label
            adapters_layout.addWidget(label)
        
        adapters_group.setLayout(adapters_layout)
        layout.addWidget(adapters_group)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        
        self.latency_label = QLabel("⏱️ Latency: --")
        self.tokens_label = QLabel("📝 Tokens: --")
        self.queries_label = QLabel("💬 Queries: 0")
        
        for label in [self.latency_label, self.tokens_label, self.queries_label]:
            label.setFont(QFont("Monaco", 10))
            metrics_layout.addWidget(label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Recent interactions log
        log_group = QGroupBox("Interaction Log")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Monaco", 9))
        self.log_display.setMaximumHeight(200)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #888;
                border: 1px solid #333;
            }
        """)
        
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel
    
    def init_roku(self):
        """Initialize Roku AI"""
        try:
            self.statusBar().showMessage("Loading Roku AI...")
            self.roku = PersonalizedRoku(
                username="Srimaan",
                use_personality_adapter=True,
                verbose=False
            )
            self.update_adapter_display()
            self.statusBar().showMessage("✅ Roku AI Ready")
            self.append_to_chat("System", "Roku AI initialized. Ready to chat!", color="#4caf50")
        except Exception as e:
            self.statusBar().showMessage(f"❌ Error: {e}")
            self.append_to_chat("System", f"Error initializing Roku: {e}", color="#f44336")
    
    def send_message(self):
        """Send a message to Roku"""
        message = self.input_field.text().strip()
        if not message:
            return
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.statusBar().showMessage("🤔 Thinking...")
        
        # Display user message
        self.append_to_chat("You", message, color="#64b5f6")
        self.input_field.clear()
        
        # Start worker thread
        self.worker = RokuWorker(self.roku, message, self.history.copy())
        self.worker.finished.connect(self.on_response)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_response(self, response: str, metadata: dict):
        """Handle response from Roku"""
        # Display response
        self.append_to_chat("Roku", response, color="#81c784")
        
        # Update history
        self.history.append({"role": "user", "content": self.history[-1] if self.history else ""})
        self.history.append({"role": "assistant", "content": response})
        
        # Update metrics
        self.update_metrics(metadata)
        
        # Log interaction
        self.log_interaction(metadata)
        
        # Save to conversation log
        self.conversation_log.append({
            "user": self.history[-2]["content"] if len(self.history) >= 2 else "",
            "assistant": response,
            "metadata": metadata
        })
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()
        self.statusBar().showMessage("✅ Ready")
    
    def on_error(self, error: str):
        """Handle error"""
        self.append_to_chat("System", f"Error: {error}", color="#f44336")
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.statusBar().showMessage("❌ Error occurred")
    
    def append_to_chat(self, sender: str, message: str, color: str = "#ffffff"):
        """Append message to chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.append(f'<span style="color: {color}; font-weight: bold;">[{timestamp}] {sender}:</span>')
        self.chat_display.append(f'<span style="color: #d4d4d4;">{message}</span>')
        self.chat_display.append("")  # Empty line
        
        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
    
    def update_adapter_display(self):
        """Update active adapter display"""
        if not self.roku:
            return
        
        active = self.roku.llm.active_adapters
        scales = self.roku.llm.adapter_info
        
        for adapter_name, label in self.adapter_labels.items():
            if adapter_name in active:
                scale = scales.get(adapter_name, 1.0)
                label.setText(f"✅ {adapter_name}: Active (scale={scale:.1f})")
                label.setStyleSheet("color: #4caf50;")
            else:
                label.setText(f"❌ {adapter_name}: Inactive")
                label.setStyleSheet("color: #757575;")
    
    def update_metrics(self, metadata: dict):
        """Update performance metrics"""
        latency = metadata.get("latency", 0)
        self.latency_label.setText(f"⏱️ Latency: {latency:.2f}s")
        
        # Update query count
        current_queries = int(self.queries_label.text().split(":")[1].strip())
        self.queries_label.setText(f"💬 Queries: {current_queries + 1}")
    
    def log_interaction(self, metadata: dict):
        """Log interaction to monitoring panel"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        adapters = ", ".join(metadata.get("adapters", []))
        latency = metadata.get("latency", 0)
        
        log_entry = f"[{timestamp}] Adapters: {adapters or 'base'} | Latency: {latency:.2f}s"
        self.log_display.append(log_entry)
    
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        self.chat_display.clear()
        self.conversation_log.clear()
        self.append_to_chat("System", "Conversation history cleared.", color="#ff9800")
        self.queries_label.setText("💬 Queries: 0")
    
    def save_conversation(self):
        """Save conversation to file"""
        if not self.conversation_log:
            self.statusBar().showMessage("No conversation to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path.home() / f"Roku/roku-ai/data/conversations/gui_conversation_{timestamp}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "interactions": self.conversation_log,
                "total_queries": len(self.conversation_log)
            }, f, indent=2)
        
        self.statusBar().showMessage(f"✅ Saved to {filepath.name}")
        self.append_to_chat("System", f"Conversation saved to {filepath.name}", color="#4caf50")


def main():
    """Launch Roku GUI"""
    app = QApplication(sys.argv)
    
    # Dark theme
    app.setStyle("Fusion")
    
    window = RokuGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
