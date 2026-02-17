import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext(null);

/**
 * Theme configurations for light and dark modes
 */
const themes = {
  light: {
    '--color-primary': '#2563eb',
    '--color-primary-dark': '#1d4ed8',
    '--color-primary-light': '#3b82f6',
    '--color-secondary': '#6b7280',
    '--color-success': '#059669',
    '--color-error': '#dc2626',
    '--color-warning': '#d97706',
    '--bg-primary': '#f8fafc',
    '--bg-secondary': '#ffffff',
    '--bg-tertiary': '#f1f5f9',
    '--bg-card': '#ffffff',
    '--text-primary': '#0f172a',
    '--text-secondary': '#475569',
    '--text-muted': '#94a3b8',
    '--border-color': '#e2e8f0',
    '--shadow-sm': '0 1px 2px rgba(0, 0, 0, 0.05)',
    '--shadow-md': '0 4px 6px rgba(0, 0, 0, 0.07)',
    '--shadow-lg': '0 10px 15px rgba(0, 0, 0, 0.1)',
  },
  dark: {
    '--color-primary': '#3b82f6',
    '--color-primary-dark': '#2563eb',
    '--color-primary-light': '#60a5fa',
    '--color-secondary': '#6b7280',
    '--color-success': '#10b981',
    '--color-error': '#ef4444',
    '--color-warning': '#f59e0b',
    '--bg-primary': '#0f172a',
    '--bg-secondary': '#1e293b',
    '--bg-tertiary': '#334155',
    '--bg-card': '#1e293b',
    '--text-primary': '#f8fafc',
    '--text-secondary': '#94a3b8',
    '--text-muted': '#64748b',
    '--border-color': '#334155',
    '--shadow-sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
    '--shadow-md': '0 4px 6px rgba(0, 0, 0, 0.3)',
    '--shadow-lg': '0 10px 15px rgba(0, 0, 0, 0.3)',
  },
};

/**
 * Apply theme CSS variables to document root
 */
const applyTheme = (themeName) => {
  const theme = themes[themeName];
  const root = document.documentElement;

  Object.entries(theme).forEach(([property, value]) => {
    root.style.setProperty(property, value);
  });

  // Also set a data attribute for any CSS that needs it
  root.setAttribute('data-theme', themeName);
};

/**
 * Theme Provider component - manages theme state
 */
export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    // Check localStorage for saved preference
    const saved = localStorage.getItem('theme');
    if (saved && (saved === 'light' || saved === 'dark')) {
      return saved;
    }
    // Default to light mode
    return 'light';
  });

  // Apply theme on mount and when theme changes
  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  /**
   * Toggle between light and dark themes
   */
  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  /**
   * Set a specific theme
   */
  const setSpecificTheme = (themeName) => {
    if (themeName === 'light' || themeName === 'dark') {
      setTheme(themeName);
    }
  };

  const value = {
    theme,
    isDark: theme === 'dark',
    isLight: theme === 'light',
    toggleTheme,
    setTheme: setSpecificTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * Hook to use theme context
 */
export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

export default ThemeContext;
