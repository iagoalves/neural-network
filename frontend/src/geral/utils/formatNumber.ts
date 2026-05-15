export function formatNumber(value: number, precision = 3): string {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-12) {
    return '0';
  }

  const roundedInteger = Math.round(value);
  if (Math.abs(value - roundedInteger) < 1e-9) {
    return String(roundedInteger);
  }

  return value
    .toFixed(precision)
    .replace(/\.?0+$/, '');
}
