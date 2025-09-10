import streamlit as st
import requests
from typing import Optional
import time
import threading
from datetime import datetime

st.set_page_config(
    page_title="Deposit Slip System", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📄"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        color: white;
    }
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .summary-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status-warning {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status-error {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .scroll-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 20px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        transition: all 0.3s ease;
    }
    .scroll-to-top:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

if 'api_base' not in st.session_state:
    st.session_state.api_base = 'https://deposite-slip-5.onrender.com'
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

def auth_headers() -> dict:
    return {}

def show_loading_spinner(message: str, type: str = "default"):
    """Show different types of loading spinners with custom messages"""
    if type == "upload":
        with st.spinner(f"📤 {message}"):
            time.sleep(0.1)
    elif type == "processing":
        with st.spinner(f"🔍 {message}"):
            time.sleep(0.1)
    elif type == "analyzing":
        with st.spinner(f"🧠 {message}"):
            time.sleep(0.1)
    elif type == "validating":
        with st.spinner(f"✅ {message}"):
            time.sleep(0.1)
    else:
        with st.spinner(f"⏳ {message}"):
            time.sleep(0.1)

def show_progress_bar(message: str, progress: float = 0.0):
    """Show a progress bar with custom message"""
    progress_bar = st.progress(progress)
    status_text = st.empty()
    status_text.text(f"🔄 {message}")
    return progress_bar, status_text

def show_animated_loading(message: str, duration: int = 3):
    """Show animated loading with multiple stages"""
    stages = [
        "📤 Uploading file...",
        "🔍 Extracting text with OCR...",
        "🧠 Analyzing content with AI...",
        "✅ Validating data...",
        "💾 Processing results..."
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, stage in enumerate(stages):
        if i < len(stages):
            progress = (i + 1) / len(stages)
            progress_bar.progress(progress)
            status_text.text(f"🔄 {stage}")
            time.sleep(duration / len(stages))
    
    return progress_bar, status_text

def show_upload_loading():
    """Show comprehensive upload loading animation"""
    # Add JavaScript to scroll to top and keep focus on loading
    st.markdown("""
    <script>
    // Scroll to top immediately
    window.scrollTo({top: 0, behavior: 'smooth'});
    
    // Keep scrolling to top every 500ms during loading
    const scrollInterval = setInterval(() => {
        window.scrollTo({top: 0, behavior: 'smooth'});
    }, 500);
    
    // Store interval ID for cleanup
    window.scrollInterval = scrollInterval;
    </script>
    """, unsafe_allow_html=True)
    
    # Create containers for different loading elements
    main_container = st.container()
    progress_container = st.container()
    status_container = st.container()
    
    with main_container:
        st.markdown("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin: 20px 0; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
            <h2>🚀 Processing Your Deposit Slip</h2>
            <p style="font-size: 18px; margin-top: 10px;">Please wait while we analyze your document...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with status_container:
        # Create columns for different status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📤 Upload Status**")
            upload_status = st.empty()
        
        with col2:
            st.markdown("**🔍 OCR Status**")
            ocr_status = st.empty()
        
        with col3:
            st.markdown("**✅ Validation Status**")
            validation_status = st.empty()
    
    return progress_bar, status_text, upload_status, ocr_status, validation_status

def format_processing_results(data):
    """Format the JSON processing results into a beautiful, readable summary"""
    
    # Extract key information from the response
    collection_id = data.get('collection_id', 'N/A')
    file_path = data.get('file_path', 'N/A')
    file_hash = data.get('file_hash', 'N/A')[:16] + "..." if data.get('file_hash') else 'N/A'
    
    # Amount information
    extracted_amount = data.get('extracted_amount')
    manual_amount = data.get('manual_amount')
    final_amount = manual_amount if manual_amount is not None else extracted_amount
    
    # Date information
    extracted_date = data.get('extracted_date')
    manual_date = data.get('manual_date')
    final_date = manual_date if manual_date is not None else extracted_date
    
    # Bank information
    bank_name = data.get('bank_name', 'Not detected')
    account_number = data.get('account_number', 'Not detected')
    
    # Validation information
    is_validated = data.get('is_validated', False)
    validation_errors = data.get('validation_errors', '')
    confidence_score = data.get('confidence_score', 0.0)
    status = data.get('status', 'unknown')
    
    # Processing details
    processing_details = data.get('processing_details', {})
    mode_used = processing_details.get('mode_used', 'unknown') if processing_details else 'unknown'
    
    # Create the formatted summary
    st.markdown("### 📊 **Processing Summary**")
    
    # Main information cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4>💰 Amount Detected</h4>
            <h2>${:,.2f}</h2>
        </div>
        """.format(final_amount if final_amount else 0), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4>📅 Date Detected</h4>
            <h3>{}</h3>
        </div>
        """.format(str(final_date) if final_date else 'Not detected'), unsafe_allow_html=True)
    
    with col3:
        status_color = "#28a745" if is_validated else "#ffc107" if status == "needs_review" else "#dc3545"
        status_icon = "✅" if is_validated else "⚠️" if status == "needs_review" else "❌"
        st.markdown("""
        <div style="background: linear-gradient(135deg, {} 0%, {} 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h4>{} Status</h4>
            <h3>{}</h3>
        </div>
        """.format(status_color, status_color, status_icon, status.title()), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed information
    st.markdown("### 🔍 **Detailed Information**")
    
    # Create expandable sections for different categories
    with st.expander("📋 **Basic Information**", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Collection ID:** {collection_id}")
            st.markdown(f"**Processing Mode:** {mode_used.upper()}")
            st.markdown(f"**Confidence Score:** {confidence_score:.1%}")
        with col2:
            st.markdown(f"**File Hash:** `{file_hash}`")
            st.markdown(f"**File Path:** `{file_path.split('/')[-1] if file_path != 'N/A' else 'N/A'}`")
    
    with st.expander("🏦 **Bank Information**"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Bank Name:** {bank_name}")
        with col2:
            st.markdown(f"**Account Number:** {account_number}")
    
    with st.expander("💰 **Amount Details**"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Extracted Amount:** ${extracted_amount:,.2f}" if extracted_amount else "**Extracted Amount:** Not detected")
        with col2:
            st.markdown(f"**Manual Override:** ${manual_amount:,.2f}" if manual_amount else "**Manual Override:** None")
        with col3:
            st.markdown(f"**Final Amount:** ${final_amount:,.2f}" if final_amount else "**Final Amount:** Not available")
    
    with st.expander("📅 **Date Details**"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Extracted Date:** {extracted_date}" if extracted_date else "**Extracted Date:** Not detected")
        with col2:
            st.markdown(f"**Manual Override:** {manual_date}" if manual_date else "**Manual Override:** None")
        with col3:
            st.markdown(f"**Final Date:** {final_date}" if final_date else "**Final Date:** Not available")
    
    # Validation section
    if validation_errors:
        with st.expander("⚠️ **Validation Issues**", expanded=True):
            st.error(f"**Issues Found:** {validation_errors}")
            st.info("💡 **Recommendation:** Please review the extracted data and consider manual overrides if needed.")
    else:
        with st.expander("✅ **Validation Results**"):
            st.success("**All validations passed!** The extracted data appears to be accurate.")
    
    # Processing details
    if processing_details:
        with st.expander("⚙️ **Processing Details**"):
            for key, value in processing_details.items():
                if value is not None:
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Action recommendations
    st.markdown("### 🎯 **Next Steps**")
    
    if status == "processed":
        st.success("🎉 **Great!** Your deposit slip has been successfully processed and validated.")
    elif status == "needs_review":
        st.warning("⚠️ **Review Required:** The system detected some issues. Please review the extracted data above.")
        st.info("💡 **You can:** 1) Accept the current data, 2) Make manual corrections, or 3) Re-upload with different settings.")
    else:
        st.error("❌ **Processing Failed:** There was an issue processing your deposit slip.")
        st.info("💡 **Try:** 1) Check your file quality, 2) Try a different processing mode, or 3) Contact support.")
    
    # Add download functionality
    st.markdown("### 📥 **Download Results**")
    
    # Create a summary report
    summary_report = f"""
DEPOSIT SLIP PROCESSING REPORT
==============================

Collection ID: {collection_id}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Processing Mode: {mode_used.upper()}
Confidence Score: {confidence_score:.1%}

EXTRACTED INFORMATION:
- Amount: ${final_amount:,.2f} (Final)
- Date: {final_date} (Final)
- Bank Name: {bank_name}
- Account Number: {account_number}

VALIDATION STATUS:
- Status: {status.upper()}
- Validated: {'Yes' if is_validated else 'No'}
- Errors: {validation_errors if validation_errors else 'None'}

PROCESSING DETAILS:
- File Hash: {file_hash}
- File Path: {file_path}
"""
    
    if processing_details:
        summary_report += "\nADDITIONAL DETAILS:\n"
        for key, value in processing_details.items():
            if value is not None:
                summary_report += f"- {key.replace('_', ' ').title()}: {value}\n"
    
    # Create download button
    st.download_button(
        label="📄 Download Processing Report",
        data=summary_report,
        file_name=f"deposit_slip_report_{collection_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        help="Download a text report of the processing results"
    )
    
    return {
        'final_amount': final_amount,
        'final_date': final_date,
        'is_validated': is_validated,
        'status': status,
        'confidence_score': confidence_score
    }

def login_view():
    st.header("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        with st.spinner("🔄 Logging in..."):
            resp = requests.post(
                f"{st.session_state.api_base}/auth/login",
                data={"email": email, "password": password},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                st.session_state.token = data["access_token"]
                st.session_state.user = data["user"]
                st.success("✅ Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error(f"❌ Login failed: {resp.text}")

def register_view():
    st.header("Register")
    with st.form("register_form"):
        email = st.text_input("Email")
        name = st.text_input("Name")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["accountant", "admin"])
        branch_id = st.number_input("Branch ID", min_value=0, step=1)
        submitted = st.form_submit_button("Register")
    if submitted:
        with st.spinner("🔄 Creating account..."):
            payload = {
                "email": email,
                "name": name,
                "password": password,
                "role": role,
                "branch_id": int(branch_id) if branch_id else None,
            }
            resp = requests.post(f"{st.session_state.api_base}/auth/register", json=payload, timeout=30)
            if resp.ok:
                st.success("✅ User created successfully! Please login.")
            else:
                st.error(f"❌ Registration failed: {resp.text}")

def collections_view():
    # Disabled for now: navigation hides this view
    st.header("Collections")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Refresh"):
            st.rerun()
    with cols[1]:
        with st.form("create_collection"):
            amount = st.number_input("Amount", min_value=0.0, step=0.01)
            date = st.date_input("Date")
            branch_id = st.number_input("Branch ID", min_value=0, step=1)
            description = st.text_input("Description", "")
            submitted = st.form_submit_button("Create Collection")
        if submitted:
            with st.spinner("🔄 Creating collection..."):
                payload = {
                    "amount": float(amount),
                    "date": str(date),
                    "branch_id": int(branch_id),
                    "description": description or None,
                }
                resp = requests.post(f"{st.session_state.api_base}/collections", json=payload, headers=auth_headers(), timeout=60)
                if resp.ok:
                    st.success("✅ Collection created successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to create collection: {resp.text}")

    with st.spinner("🔄 Loading collections..."):
        try:
            resp = requests.get(
                f"{st.session_state.api_base}/collections",
                headers=auth_headers(),
                timeout=(10, 180),  # connect, read
            )
            if resp.ok:
                data = resp.json()
                if data:
                    st.dataframe(data, use_container_width=True)
                else:
                    st.info("📭 No collections found. Create your first collection above!")
            else:
                st.error(f"❌ Failed to load collections: {resp.text}")
        except requests.Timeout:
            st.warning("⏰ API timed out. If the server is cold-starting (Render), try again in ~30-60s.")
        except requests.RequestException as e:
            st.error(f"❌ Request failed: {e}")

def upload_view():
    st.header("📄 Upload Deposit Slip")
    
    # Add some styling
    st.markdown("""
    <style>
    .upload-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .upload-header {
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.form("upload_form"):
        st.markdown('<div class="upload-container"><div class="upload-header"><h3>🚀 Upload Your Deposit Slip</h3></div></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            collection_id = st.number_input("Collection ID", min_value=1, step=1, help="Enter the collection ID for this deposit slip")
            manual_amount = st.number_input("Manual Amount (Optional)", min_value=0.0, step=0.01, help="Override the detected amount if needed")
        
        with col2:
            manual_date = st.date_input("Manual Date (Optional)", value=None, help="Override the detected date if needed")
            mode = st.selectbox("Processing Mode", ["vision", "gcv"], index=0, help="Choose the AI processing method")
        
        file = st.file_uploader(
            "📎 Deposit slip file (image or PDF)", 
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
            help="Upload an image or PDF of your deposit slip"
        )
        
        # Enhanced submit button
        submitted = st.form_submit_button(
            "🚀 Upload & Process", 
            type="primary",
            use_container_width=True
        )
    
    if submitted:
        if not file:
            st.warning("⚠️ Please choose a file to upload")
            return
        
        # Scroll to top to show loading animation
        st.markdown("""
        <script>
        window.scrollTo(0, 0);
        </script>
        """, unsafe_allow_html=True)
        
        # Show comprehensive loading animation
        progress_bar, status_text, upload_status, ocr_status, validation_status = show_upload_loading()
        
        try:
            # Stage 1: File preparation
            upload_status.success("✅ File ready")
            progress_bar.progress(0.1)
            status_text.text("📤 Preparing file for upload...")
            time.sleep(0.5)
            
            files = {"file": (file.name, file.getvalue(), file.type)}
            data = {
                "collection_id": str(int(collection_id)),
                "manual_amount": str(float(manual_amount)) if manual_amount else None,
                "manual_date": str(manual_date) if manual_date else None,
                "mode": mode,
            }
            # Remove None values for multipart
            data = {k: v for k, v in data.items() if v not in (None, "None", "")}
            
            # Stage 2: Upload
            upload_status.info("📤 Uploading...")
            progress_bar.progress(0.2)
            status_text.text("📤 Uploading file to server...")
            time.sleep(0.5)
            
            # Stage 3: OCR Processing
            ocr_status.info("🔍 Processing...")
            progress_bar.progress(0.3)
            status_text.text("🔍 Extracting text with OCR...")
            time.sleep(0.5)
            
            # Stage 4: AI Analysis
            progress_bar.progress(0.4)
            status_text.text("🧠 Analyzing content with AI...")
            time.sleep(0.5)
            
            # Stage 5: Validation
            validation_status.info("✅ Validating...")
            progress_bar.progress(0.5)
            status_text.text("✅ Validating extracted data...")
            time.sleep(0.5)
            
            # Make the actual request
            progress_bar.progress(0.6)
            status_text.text("🌐 Sending request to server...")
            
            resp = requests.post(
                f"{st.session_state.api_base}/deposit-slips/upload",
                files=files,
                data=data,
                headers=auth_headers(),
                timeout=600,
            )
            
            # Update progress based on response
            if resp.ok:
                progress_bar.progress(1.0)
                status_text.text("🎉 Processing complete!")
                upload_status.success("✅ Upload complete")
                ocr_status.success("✅ OCR complete")
                validation_status.success("✅ Validation complete")
                
                time.sleep(1)
                
                # Clear loading elements and stop auto-scroll
                progress_bar.empty()
                status_text.empty()
                upload_status.empty()
                ocr_status.empty()
                validation_status.empty()
                
                # Stop the auto-scroll interval
                st.markdown("""
                <script>
                if (window.scrollInterval) {
                    clearInterval(window.scrollInterval);
                    window.scrollInterval = null;
                }
                </script>
                """, unsafe_allow_html=True)
                
                # Show success message with styling
                st.success("🎉 **Upload Successful!** Your deposit slip has been processed.")
                
                # Display formatted results
                result_data = resp.json()
                summary = format_processing_results(result_data)
                
                # Display raw JSON in an expandable section for technical users
                with st.expander("🔧 **Raw JSON Data (For Technical Users)**"):
                    st.json(result_data)
                
                # Add some additional info based on the results
                if summary['status'] == 'processed':
                    st.info("💡 **Tip:** Your deposit slip has been successfully processed and is ready for use!")
                elif summary['status'] == 'needs_review':
                    st.warning("💡 **Tip:** Please review the extracted data above. You may need to make corrections or re-upload.")
                else:
                    st.info("💡 **Tip:** You can download the processed data or make corrections if needed.")
                
            else:
                progress_bar.progress(0.0)
                status_text.text("❌ Processing failed")
                upload_status.error("❌ Upload failed")
                ocr_status.error("❌ Processing failed")
                validation_status.error("❌ Validation failed")
                
                time.sleep(1)
                
                # Clear loading elements and stop auto-scroll
                progress_bar.empty()
                status_text.empty()
                upload_status.empty()
                ocr_status.empty()
                validation_status.empty()
                
                # Stop the auto-scroll interval
                st.markdown("""
                <script>
                if (window.scrollInterval) {
                    clearInterval(window.scrollInterval);
                    window.scrollInterval = null;
                }
                </script>
                """, unsafe_allow_html=True)
                
                st.error(f"❌ **Upload Failed:** {resp.text}")
                
        except requests.Timeout:
            progress_bar.progress(0.0)
            status_text.text("⏰ Request timed out")
            upload_status.error("⏰ Timeout")
            ocr_status.error("⏰ Timeout")
            validation_status.error("⏰ Timeout")
            
            time.sleep(1)
            
            # Clear loading elements and stop auto-scroll
            progress_bar.empty()
            status_text.empty()
            upload_status.empty()
            ocr_status.empty()
            validation_status.empty()
            
            # Stop the auto-scroll interval
            st.markdown("""
            <script>
            if (window.scrollInterval) {
                clearInterval(window.scrollInterval);
                window.scrollInterval = null;
            }
            </script>
            """, unsafe_allow_html=True)
            
            st.warning("⏰ **Request Timeout:** The server took too long to respond. This might be due to heavy processing. Please try again.")
            
        except requests.RequestException as e:
            progress_bar.progress(0.0)
            status_text.text("❌ Connection error")
            upload_status.error("❌ Connection failed")
            ocr_status.error("❌ Connection failed")
            validation_status.error("❌ Connection failed")
            
            time.sleep(1)
            
            # Clear loading elements and stop auto-scroll
            progress_bar.empty()
            status_text.empty()
            upload_status.empty()
            ocr_status.empty()
            validation_status.empty()
            
            # Stop the auto-scroll interval
            st.markdown("""
            <script>
            if (window.scrollInterval) {
                clearInterval(window.scrollInterval);
                window.scrollInterval = null;
            }
            </script>
            """, unsafe_allow_html=True)
            
            st.error(f"❌ **Connection Error:** {str(e)}")
            
        except Exception as e:
            progress_bar.progress(0.0)
            status_text.text("❌ Unexpected error")
            upload_status.error("❌ Error")
            ocr_status.error("❌ Error")
            validation_status.error("❌ Error")
            
            time.sleep(1)
            
            # Clear loading elements and stop auto-scroll
            progress_bar.empty()
            status_text.empty()
            upload_status.empty()
            ocr_status.empty()
            validation_status.empty()
            
            # Stop the auto-scroll interval
            st.markdown("""
            <script>
            if (window.scrollInterval) {
                clearInterval(window.scrollInterval);
                window.scrollInterval = null;
            }
            </script>
            """, unsafe_allow_html=True)
            
            st.error(f"❌ **Unexpected Error:** {str(e)}")

def slips_view():
    # Disabled for now: navigation hides this view
    st.header("Deposit Slips")
    status = st.selectbox("Filter by status", ["", "pending", "needs_review", "processed"]) or None
    params = {}
    if status:
        params["status"] = status
    with st.spinner("🔄 Loading deposit slips..."):
        try:
            resp = requests.get(
                f"{st.session_state.api_base}/deposit-slips",
                headers=auth_headers(),
                params=params,
                timeout=(10, 180),  # connect, read
            )
            if resp.ok:
                data = resp.json()
                if data:
                    st.dataframe(data, use_container_width=True)
                    # Simple override UI for slips needing review
                    ids = [row["id"] for row in data if row.get("status") == "needs_review"]
                    if ids:
                        st.subheader("🔧 Record Override (requires reason & approver)")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            slip_id = st.selectbox("Select Slip", ids)
                        with col2:
                            reason = st.text_input("Override Reason")
                        with col3:
                            approver = st.text_input("Approved by")
                        if st.button("✅ Submit Override", type="primary"):
                            with st.spinner("🔄 Processing override..."):
                                form = {"reason": reason, "approved_by": approver}
                                try:
                                    r = requests.post(
                                        f"{st.session_state.api_base}/deposit-slips/{slip_id}/override",
                                        data=form,
                                        timeout=(10, 180),
                                    )
                                    if r.ok:
                                        st.success("✅ Override recorded successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Override failed: {r.text}")
                                except requests.Timeout:
                                    st.warning("⏰ Override request timed out. Please retry.")
                                except requests.RequestException as e:
                                    st.error(f"❌ Override request failed: {e}")
                else:
                    st.info("📭 No deposit slips found. Upload your first deposit slip!")
            else:
                st.error(f"❌ Failed to load deposit slips: {resp.text}")
        except requests.Timeout:
            st.warning("⏰ API timed out. If the server is cold-starting (Render), try again in ~30-60s.")
        except requests.RequestException as e:
            st.error(f"❌ Request failed: {e}")

def topbar():
    with st.sidebar:
        st.text_input("API Base URL", key="api_base")
        st.markdown("---")
        # Show only Upload for now
        view = st.radio("Navigate", ["Upload"])
    return view

def main():
    # Add main header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Deposit Slip Processing System</h1>
        <p>AI-Powered Document Analysis & Validation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add scroll-to-top button
    st.markdown("""
    <button class="scroll-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})" title="Scroll to top">
        ↑
    </button>
    """, unsafe_allow_html=True)
    
    view = topbar()
    if view == "Upload":
        upload_view()
    # Other views are disabled

if __name__ == "__main__":
    main()


