-- Demo database schema and seed data for SQLTest Pro
-- Creates users, orders tables and user_orders view with intentionally problematic data

-- Users table with validation issues
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT,
    age INTEGER,
    name TEXT,
    status TEXT,
    created_at TEXT,
    salary REAL
);

-- Orders table
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    product TEXT,
    amount REAL,
    order_date TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- User orders view
CREATE VIEW user_orders AS
SELECT 
    u.id AS user_id,
    u.name AS user_name,
    u.email,
    o.id AS order_id,
    o.product,
    o.amount,
    o.order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Insert sample users data (with validation issues for demo)
INSERT INTO users (id, email, age, name, status, created_at, salary) VALUES
    (1, 'alice@example.com', 28, 'Alice Johnson', 'active', '2023-01-15', 75000.00),
    (2, 'bob.invalid-email', 34, 'Bob Smith', 'inactive', '2023-02-20', 65000.00),  -- Invalid email
    (3, 'charlie@test.org', 150, 'Charlie Brown', 'active', '2023-03-10', 80000.00),  -- Invalid age
    (4, NULL, 25, 'David Wilson', 'pending', '2023-04-05', 70000.00),  -- NULL email
    (5, 'eve@company.com', -5, 'Eve Davis', 'active', '2023-05-12', 90000.00),  -- Negative age
    (6, 'frank@domain.co', 42, 'F', 'invalid_status', '2023-06-18', 55000.00),  -- Short name, invalid status
    (7, 'grace@email.net', 29, 'Grace Lee with a very long name that exceeds normal limits', 'active', '2023-07-22', 85000.00),  -- Long name
    (8, 'henry@test.com', 31, 'Henry Garcia', 'inactive', '2023-08-30', NULL),  -- NULL salary
    (9, 'isabel@example.org', 26, 'Isabel Martinez', 'pending', '2023-09-14', -1000.00),  -- Negative salary
    (10, 'jack@company.com', 38, 'Jack Taylor', 'active', '2023-10-08', 95000.00);

-- Insert sample orders data
INSERT INTO orders (id, user_id, product, amount, order_date) VALUES
    (1, 1, 'Laptop', 1299.99, '2023-11-01'),
    (2, 1, 'Mouse', 29.99, '2023-11-02'),
    (3, 2, 'Keyboard', 79.99, '2023-11-03'),
    (4, 3, 'Monitor', 399.99, '2023-11-04'),
    (5, 4, 'Headphones', 199.99, '2023-11-05'),
    (6, 6, 'Tablet', 599.99, '2023-11-06'),
    (7, 7, 'Phone', 899.99, '2023-11-07'),
    (8, 8, 'Charger', 39.99, '2023-11-08'),
    (9, 10, 'Webcam', 129.99, '2023-11-09'),
    (10, 10, 'Speakers', 149.99, '2023-11-10');
