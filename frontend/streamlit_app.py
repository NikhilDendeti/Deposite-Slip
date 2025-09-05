import streamlit as st
import requests
from typing import Optional

st.set_page_config(page_title="Deposit Slip System", layout="wide")

if 'api_base' not in st.session_state:
    st.session_state.api_base = 'https://deposite-slip-5.onrender.com'
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

def auth_headers() -> dict:
    return {}

def login_view():
    st.header("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        resp = requests.post(
            f"{st.session_state.api_base}/auth/login",
            data={"email": email, "password": password},
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            st.session_state.token = data["access_token"]
            st.session_state.user = data["user"]
            st.success("Logged in")
            st.experimental_rerun()
        else:
            st.error(resp.text)

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
        payload = {
            "email": email,
            "name": name,
            "password": password,
            "role": role,
            "branch_id": int(branch_id) if branch_id else None,
        }
        resp = requests.post(f"{st.session_state.api_base}/auth/register", json=payload, timeout=30)
        if resp.ok:
            st.success("User created. Please login.")
        else:
            st.error(resp.text)

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
            payload = {
                "amount": float(amount),
                "date": str(date),
                "branch_id": int(branch_id),
                "description": description or None,
            }
            resp = requests.post(f"{st.session_state.api_base}/collections", json=payload, headers=auth_headers(), timeout=60)
            if resp.ok:
                st.success("Collection created")
                st.rerun()
            else:
                st.error(resp.text)

    try:
        resp = requests.get(
            f"{st.session_state.api_base}/collections",
            headers=auth_headers(),
            timeout=(10, 180),  # connect, read
        )
        if resp.ok:
            data = resp.json()
            st.dataframe(data, use_container_width=True)
        else:
            st.error(resp.text)
    except requests.Timeout:
        st.warning("API timed out. If the server is cold-starting (Render), try again in ~30-60s.")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

def upload_view():
    st.header("Upload Deposit Slip")
    with st.form("upload_form"):
        collection_id = st.number_input("Collection ID", min_value=1, step=1)
        manual_amount = st.number_input("Manual Amount", min_value=0.0, step=0.01)
        manual_date = st.date_input("Manual Date", value=None)
        file = st.file_uploader("Deposit slip file (image or PDF)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"])
        # Support OCR, Vision and Google Cloud Vision modes
        # removed ocr mode
        mode = st.selectbox("Processing mode", ["vision", "gcv"], index=0)
        submitted = st.form_submit_button("Upload")
    if submitted:
        if not file:
            st.warning("Please choose a file")
            return
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {
            "collection_id": str(int(collection_id)),
            "manual_amount": str(float(manual_amount)) if manual_amount else None,
            "manual_date": str(manual_date) if manual_date else None,
            "mode": mode,
        }
        # Remove None values for multipart
        data = {k: v for k, v in data.items() if v not in (None, "None", "")}
        # Increase timeout for heavy OCR/LLM processing
        resp = requests.post(
            f"{st.session_state.api_base}/deposit-slips/upload",
            files=files,
            data=data,
            headers=auth_headers(),
            timeout=600,
        )
        if resp.ok:
            st.success("Uploaded and processed")
            st.json(resp.json())
        else:
            st.error(resp.text)

def slips_view():
    # Disabled for now: navigation hides this view
    st.header("Deposit Slips")
    status = st.selectbox("Filter by status", ["", "pending", "needs_review", "processed"]) or None
    params = {}
    if status:
        params["status"] = status
    try:
        resp = requests.get(
            f"{st.session_state.api_base}/deposit-slips",
            headers=auth_headers(),
            params=params,
            timeout=(10, 180),  # connect, read
        )
        if resp.ok:
            data = resp.json()
            st.dataframe(data, use_container_width=True)
            # Simple override UI for slips needing review
            ids = [row["id"] for row in data if row.get("status") == "needs_review"]
            if ids:
                st.subheader("Record Override (requires reason & approver)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    slip_id = st.selectbox("Slip", ids)
                with col2:
                    reason = st.text_input("Reason")
                with col3:
                    approver = st.text_input("Approved by")
                if st.button("Submit Override"):
                    form = {"reason": reason, "approved_by": approver}
                    try:
                        r = requests.post(
                            f"{st.session_state.api_base}/deposit-slips/{slip_id}/override",
                            data=form,
                            timeout=(10, 180),
                        )
                        if r.ok:
                            st.success("Override recorded")
                            st.rerun()
                        else:
                            st.error(r.text)
                    except requests.Timeout:
                        st.warning("Override request timed out. Please retry.")
                    except requests.RequestException as e:
                        st.error(f"Override request failed: {e}")
        else:
            st.error(resp.text)
    except requests.Timeout:
        st.warning("API timed out. If the server is cold-starting (Render), try again in ~30-60s.")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

def topbar():
    with st.sidebar:
        st.text_input("API Base URL", key="api_base")
        st.markdown("---")
        # Show only Upload for now
        view = st.radio("Navigate", ["Upload"])
    return view

def main():
    view = topbar()
    if view == "Upload":
        upload_view()
    # Other views are disabled

if __name__ == "__main__":
    main()


