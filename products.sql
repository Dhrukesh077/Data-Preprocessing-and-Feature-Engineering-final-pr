-- products.sql
-- SQL script to create 'products' table and insert sample product rows

DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    price INTEGER,
    stock INTEGER
);

INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P001', 'Earphones', 'Electronics', 499, 50);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P002', 'Bluetooth Speaker', 'Audio', 699, 40);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P003', 'Smart Watch', 'Wearable', 1299, 25);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P004', 'Keyboard', 'Computer', 899, 60);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P005', 'Headphones', 'Audio', 1999, 30);
INSERT INTO products (product_id, product_name, category, price, stock) VALUES ('P006', 'Power Bank', 'Mobile Accessories', 2599, 20);
