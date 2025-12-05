"""
Simple Attendance System with MongoDB (without face_recognition)
Uses basic image comparison for demo purposes
"""
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pymongo
from pymongo import MongoClient
import io
import base64

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "attendance_system"

print("=" * 70)
print("EMPLOYEE ATTENDANCE SYSTEM (Simple Version)")
print("=" * 70)

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[DB_NAME]
    employees_collection = db["employees"]
    attendance_collection = db["attendance"]
    print("\n[OK] Connected to MongoDB")
    print(f"    Database: {DB_NAME}")
    print(f"    Collections: employees, attendance")
except Exception as e:
    print(f"\n[ERROR] MongoDB connection failed: {str(e)}")
    print("        Please start MongoDB server first!")
    client = None

print("=" * 70)

def image_to_base64(image):
    """Convert PIL Image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_str):
    """Convert base64 to PIL Image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def compare_images_simple(img1, img2):
    """Simple image similarity using histogram comparison"""
    # Convert to numpy arrays
    arr1 = np.array(img1.resize((128, 128)))
    arr2 = np.array(img2.resize((128, 128)))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    
    # Calculate histogram correlation
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation

def register_employee(name, employee_id, image):
    """Register new employee"""
    if client is None:
        return "[ERROR] MongoDB not connected!"
    
    if not name or not employee_id or image is None:
        return "[ERROR] Please provide name, ID, and photo"
    
    try:
        # Check if employee ID exists
        existing = employees_collection.find_one({"employee_id": employee_id})
        if existing:
            return f"[ERROR] Employee ID '{employee_id}' already exists!"
        
        # Store in MongoDB
        employee_data = {
            "name": name,
            "employee_id": employee_id,
            "face_image": image_to_base64(image),
            "registered_at": datetime.now().isoformat()
        }
        
        result = employees_collection.insert_one(employee_data)
        
        return f"""[SUCCESS] Employee registered!

Name: {name}
Employee ID: {employee_id}
Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Database ID: {result.inserted_id}
"""
        
    except Exception as e:
        return f"[ERROR] {str(e)}"

def check_attendance(image):
    """Check attendance by comparing with registered employees"""
    if client is None:
        return "[ERROR] MongoDB not connected!", None
    
    if image is None:
        return "[ERROR] Please provide an image", None
    
    try:
        # Get all employees
        employees = list(employees_collection.find())
        if len(employees) == 0:
            return "[ERROR] No employees registered in database", None
        
        # Compare with each employee
        best_match = None
        best_score = 0
        
        for employee in employees:
            emp_image = base64_to_image(employee["face_image"])
            score = compare_images_simple(image, emp_image)
            
            if score > best_score:
                best_score = score
                best_match = employee
        
        # Threshold for match (0.7 = 70% similarity)
        threshold = 0.7
        
        if best_score >= threshold:
            name = best_match["name"]
            emp_id = best_match["employee_id"]
            confidence = best_score * 100
            
            # Record attendance
            attendance_record = {
                "employee_id": emp_id,
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence,
                "similarity_score": float(best_score)
            }
            attendance_collection.insert_one(attendance_record)
            
            # Get employee photo
            employee_image = base64_to_image(best_match["face_image"])
            
            result_text = f"""[MATCH FOUND]

Name: {name}
Employee ID: {emp_id}
Confidence: {confidence:.2f}%
Similarity Score: {best_score:.4f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Attendance recorded successfully!
"""
            
            return result_text.strip(), employee_image
        else:
            return f"[NO MATCH] No matching employee found\nBest score: {best_score:.4f} (Threshold: {threshold})", None
            
    except Exception as e:
        return f"[ERROR] {str(e)}", None

def get_all_employees():
    """Get list of registered employees"""
    if client is None:
        return "[ERROR] MongoDB not connected!"
    
    try:
        employees = list(employees_collection.find())
        if len(employees) == 0:
            return "No employees registered yet"
        
        result = f"TOTAL EMPLOYEES: {len(employees)}\n\n"
        for i, emp in enumerate(employees, 1):
            result += f"{i}. Name: {emp['name']}\n"
            result += f"   ID: {emp['employee_id']}\n"
            result += f"   Registered: {emp['registered_at']}\n\n"
        
        return result
        
    except Exception as e:
        return f"[ERROR] {str(e)}"

def get_attendance_records():
    """Get recent attendance records"""
    if client is None:
        return "[ERROR] MongoDB not connected!"
    
    try:
        records = list(attendance_collection.find().sort("timestamp", pymongo.DESCENDING).limit(20))
        if len(records) == 0:
            return "No attendance records yet"
        
        result = f"RECENT ATTENDANCE (Last 20 records)\n\n"
        for i, record in enumerate(records, 1):
            result += f"{i}. {record['name']} (ID: {record['employee_id']})\n"
            result += f"   Time: {record['timestamp']}\n"
            result += f"   Confidence: {record['confidence']:.2f}%\n\n"
        
        return result
        
    except Exception as e:
        return f"[ERROR] {str(e)}"

def delete_employee(employee_id):
    """Delete employee by ID"""
    if client is None:
        return "[ERROR] MongoDB not connected!"
    
    if not employee_id:
        return "[ERROR] Please enter Employee ID"
    
    try:
        result = employees_collection.delete_one({"employee_id": employee_id})
        if result.deleted_count > 0:
            return f"[SUCCESS] Deleted employee ID: {employee_id}"
        else:
            return f"[ERROR] Employee ID '{employee_id}' not found"
    except Exception as e:
        return f"[ERROR] {str(e)}"

# Gradio Interface
with gr.Blocks(title="Attendance System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Employee Attendance System
    
    ### Simple version using image histogram comparison
    - MongoDB database for storage
    - No complex face recognition library needed
    - Works with any face photos
    """)
    
    with gr.Tabs():
        # Register Employee
        with gr.Tab("Register Employee"):
            with gr.Row():
                with gr.Column():
                    reg_name = gr.Textbox(label="Full Name", placeholder="Enter name")
                    reg_id = gr.Textbox(label="Employee ID", placeholder="EMP001")
                    reg_image = gr.Image(type="pil", label="Upload Photo", height=300)
                    reg_button = gr.Button("Register Employee", variant="primary")
                with gr.Column():
                    reg_output = gr.Textbox(label="Result", lines=12)
            
            reg_button.click(register_employee, [reg_name, reg_id, reg_image], reg_output)
        
        # Check Attendance
        with gr.Tab("Check Attendance"):
            with gr.Row():
                with gr.Column():
                    check_image = gr.Image(type="pil", label="Capture/Upload Photo", height=400)
                    check_button = gr.Button("Check Attendance", variant="primary")
                with gr.Column():
                    check_output = gr.Textbox(label="Result", lines=15)
                    matched_image = gr.Image(label="Matched Employee", height=300)
            
            check_button.click(check_attendance, check_image, [check_output, matched_image])
        
        # View Employees
        with gr.Tab("View Employees"):
            with gr.Row():
                with gr.Column():
                    refresh_emp = gr.Button("Refresh List", variant="secondary")
                    delete_id = gr.Textbox(label="Delete by ID", placeholder="Enter Employee ID")
                    delete_btn = gr.Button("Delete Employee", variant="stop")
                with gr.Column():
                    emp_list = gr.Textbox(label="Employee List", lines=20)
                    delete_result = gr.Textbox(label="Delete Result", lines=3)
            
            refresh_emp.click(get_all_employees, [], emp_list)
            delete_btn.click(delete_employee, delete_id, delete_result)
        
        # Attendance Records
        with gr.Tab("Attendance Records"):
            refresh_att = gr.Button("Refresh Records", variant="secondary")
            att_records = gr.Textbox(label="Attendance History", lines=20)
            
            refresh_att.click(get_attendance_records, [], att_records)
    
    gr.Markdown("""
    ---
    ### Setup:
    1. Start MongoDB: `mongod --dbpath C:\\data\\db`
    2. Register employees with clear photos
    3. Check attendance via webcam or upload
    
    ### Database:
    - URI: mongodb://localhost:27017/
    - Database: attendance_system
    - Collections: employees, attendance
    
    ### Note:
    This is a simplified version using histogram comparison.
    For production use, consider face_recognition library with proper face detection.
    """)

if __name__ == "__main__":
    print("\nStarting Attendance System...")
    print("URL: http://127.0.0.1:7861")
    print("Press Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
