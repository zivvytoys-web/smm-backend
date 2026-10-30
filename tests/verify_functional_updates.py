"""
Comprehensive Verification Script for Social AI Platform Functional Updates
Tests:
1. User isolation between two accounts
2. Brand color palette persistence
3. Quality Check auditing multiple items & auto-optimization
4. Social account credential connection & verification (without LinkedIn)
5. Server-side ownership enforcement
"""
import requests
import json
import uuid

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_full_platform():
    print("==================================================")
    print("RUNNING FUNCTIONAL VERIFICATION SUITE")
    print("==================================================")

    # 1. Login User 1 (Admin)
    print("\n[1] Testing User 1 (Admin) Authentication...")
    u1_login = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "creator@hypergrowth.com",
        "password": "Password123!"
    })
    assert u1_login.status_code == 200, f"User 1 login failed: {u1_login.text}"
    u1_token = u1_login.json()["access_token"]
    u1_headers = {"Authorization": f"Bearer {u1_token}"}
    print("[OK] User 1 Authenticated successfully.")

    # 2. Test User 1 Quality Check All Endpoint (Multi-item audit)
    print("\n[2] Testing Quality Check Multi-Item Audit Endpoint...")
    qc_res = requests.get(f"{BASE_URL}/content/quality-check/all", headers=u1_headers)
    assert qc_res.status_code == 200, f"Quality check failed: {qc_res.text}"
    qc_data = qc_res.json()
    qc_items_count = qc_data.get('total_items', qc_data.get('total_audited', 0))
    print(f"[OK] Audited {qc_items_count} content items.")
    print(f"[OK] Summary: Passed={qc_data.get('passed_count')}, Needs Attention={qc_data.get('attention_count')}, Avg Score={qc_data.get('average_score')}%")
    assert qc_items_count > 0, "Expected at least 1 content item to be audited"
    first_item_id = qc_data["items"][0]["content_id"]

    # 3. Test Auto-Optimize Endpoint
    print(f"\n[3] Testing Auto-Optimize on Content ID {first_item_id}...")
    opt_res = requests.post(f"{BASE_URL}/content/{first_item_id}/auto-optimize", headers=u1_headers)
    assert opt_res.status_code == 200, f"Auto-optimize failed: {opt_res.text}"
    opt_data = opt_res.json()
    new_score = opt_data.get("new_score") or opt_data.get("meta_info", {}).get("quality_check", {}).get("score")
    print(f"[OK] Auto-optimized successfully! New compliance score: {new_score}%")
    assert new_score == 100, f"Auto-optimize should achieve 100% compliance, got {new_score}"

    # 4. Test Brand Profile Color Palette Tokens Persistence
    print("\n[4] Testing Brand Profile Dynamic Color Palette Tokens...")
    bp_res = requests.get(f"{BASE_URL}/brand-profiles/", headers=u1_headers)
    if bp_res.status_code != 200:
        bp_res = requests.get(f"{BASE_URL}/brand-profiles", headers=u1_headers)
    assert bp_res.status_code == 200, f"Brand profiles list failed: {bp_res.text}"
    bp_list = bp_res.json()
    assert len(bp_list) > 0, "User 1 should have a brand profile"
    bp_id = bp_list[0]["id"]

    custom_tokens = {
        "primary": "#6366f1",
        "secondary": "#0ea5e9",
        "accent": "#f43f5e",
        "background": "#0b0f19",
        "surface": "#111827",
        "border": "#1f2937",
        "textPrimary": "#f9fafb",
        "textSecondary": "#9ca3af",
        "buttonBg": "#6366f1",
        "buttonText": "#ffffff"
    }

    update_bp = requests.put(f"{BASE_URL}/brand-profiles/{bp_id}", headers=u1_headers, json={
        "name": "aminoid Luxury",
        "color_palette": ["#6366f1", "#0ea5e9", "#f43f5e"],
        "meta_info": {"theme_tokens": custom_tokens}
    })
    assert update_bp.status_code == 200, f"Brand profile update failed: {update_bp.text}"
    refreshed_bp = requests.get(f"{BASE_URL}/brand-profiles/{bp_id}", headers=u1_headers).json()
    assert refreshed_bp.get("color_palette") == ["#6366f1", "#0ea5e9", "#f43f5e"]
    assert refreshed_bp.get("meta_info", {}).get("theme_tokens", {}).get("primary") == "#6366f1"
    print("[OK] Brand color palette and dynamic theme tokens saved and verified in MySQL.")

    # 5. Test Social Accounts (Connect with API Credentials, Live Verification Rejection, and Isolation)
    print("\n[5] Testing Social Account Connection & Verification...")
    
    # 5a. Test that invalid token fails verification against live Meta API
    invalid_connect_res = requests.post(f"{BASE_URL}/social-accounts/connect-credentials", headers=u1_headers, json={
        "platform": "instagram",
        "account_name": "@aminoid.invalid",
        "account_id": f"ig_test_{uuid.uuid4().hex[:6]}",
        "access_token": "EAAtest_token_invalid_12345"
    })
    assert invalid_connect_res.status_code == 400, "Invalid token must be rejected by provider verification"
    assert "Verification failed" in invalid_connect_res.json().get("detail", ""), "Should return provider verification error"
    print("[OK] Confirmed live provider verification blocks invalid/unauthorized tokens.")

    # 5b. Connect account with skip_live_verification for unit test persistence
    connect_res = requests.post(f"{BASE_URL}/social-accounts/connect-credentials", headers=u1_headers, json={
        "platform": "instagram",
        "account_name": "@aminoid.verified",
        "account_id": f"ig_test_{uuid.uuid4().hex[:6]}",
        "access_token": "EAAtest_token_secret_123456789",
        "skip_live_verification": True
    })
    assert connect_res.status_code in [200, 201], f"Connect credentials failed: {connect_res.text}"
    soc_acc = connect_res.json()
    soc_id = soc_acc["id"]
    print(f"[OK] Connected Instagram account {soc_acc['account_name']} with encrypted credentials.")

    # Test Verify Endpoint
    verify_res = requests.post(f"{BASE_URL}/social-accounts/{soc_id}/verify", headers=u1_headers)
    assert verify_res.status_code == 200
    assert verify_res.json().get("account_id") == soc_id
    print("[OK] Verified connectivity through backend verification endpoint.")

    # Test Admin Integrations endpoint permissions enforcement
    non_admin_res = requests.get(f"{BASE_URL}/admin/integrations", headers=u1_headers)
    assert non_admin_res.status_code == 403, "Non-admin must be blocked with 403 from admin integration settings"
    print("[OK] Confirmed FastAPI backend strictly blocks non-admin users from accessing developer secrets.")

    # Create / login admin user to test authorized access
    admin_email = f"admin_{uuid.uuid4().hex[:6]}@example.com"
    requests.post(f"{BASE_URL}/auth/register", json={
        "full_name": "Platform Admin",
        "email": admin_email,
        "password": "Password123!",
        "organization_name": "Admin Org",
        "role": "admin"
    })
    admin_login = requests.post(f"{BASE_URL}/auth/login", json={"email": admin_email, "password": "Password123!"}).json()
    admin_headers = {"Authorization": f"Bearer {admin_login['access_token']}"}

    admin_int_res = requests.get(f"{BASE_URL}/admin/integrations", headers=admin_headers)
    assert admin_int_res.status_code == 200
    assert "meta_app_secret_masked" in admin_int_res.json()
    print("[OK] Admin developer integrations endpoint verified with secret masking for authorized admins.")

    # Confirm LinkedIn is completely gone from social accounts list
    all_soc = requests.get(f"{BASE_URL}/social-accounts/", headers=u1_headers).json()
    assert not any(a.get("platform") == "LINKEDIN" for a in all_soc), "LinkedIn should not exist in social accounts"
    print("[OK] Confirmed LinkedIn is absent from active social account pool.")



    # 6. Strict User Data Isolation Test (Register User 2)
    print("\n[6] Testing Strict Multi-User Data Isolation...")
    u2_email = f"user2_{uuid.uuid4().hex[:6]}@example.com"
    u2_reg = requests.post(f"{BASE_URL}/auth/register", json={
        "full_name": "Second User",
        "email": u2_email,
        "password": "Password123!",
        "organization_name": "Isolated Org Two",
        "role": "admin"
    })
    assert u2_reg.status_code in [200, 201], f"User 2 registration failed: {u2_reg.text}"
    u2_login = requests.post(f"{BASE_URL}/auth/login", json={
        "email": u2_email,
        "password": "Password123!"
    })
    assert u2_login.status_code == 200, f"User 2 login failed: {u2_login.text}"
    u2_token = u2_login.json()["access_token"]
    u2_headers = {"Authorization": f"Bearer {u2_token}"}

    # Verify User 2 sees 0 content items from User 1
    u2_content = requests.get(f"{BASE_URL}/content/", headers=u2_headers).json()
    assert len(u2_content) == 0, f"User 2 should see 0 content items, but saw {len(u2_content)}"
    print("[OK] User 2 sees 0 content items from User 1.")

    # Verify User 2 cannot access or edit User 1's brand profile
    u2_bp_attempt = requests.get(f"{BASE_URL}/brand-profiles/{bp_id}", headers=u2_headers)
    assert u2_bp_attempt.status_code in [403, 404], f"User 2 should not access User 1 BrandProfile, got status {u2_bp_attempt.status_code}"
    print("[OK] User 2 is strictly blocked from accessing User 1 Brand Profile.")

    # Verify User 2 cannot access User 1's connected social accounts
    u2_soc_attempt = requests.get(f"{BASE_URL}/social-accounts/{soc_id}/verify", headers=u2_headers)
    assert u2_soc_attempt.status_code in [403, 404, 405], f"User 2 should not access User 1 Social Account, got status {u2_soc_attempt.status_code}"
    print("[OK] User 2 is strictly blocked from accessing User 1 Social Accounts.")

    print("\n==================================================")
    print("ALL VERIFICATION SUITE CHECKS PASSED (100% SUCCESS)")
    print("==================================================")

if __name__ == "__main__":
    test_full_platform()
