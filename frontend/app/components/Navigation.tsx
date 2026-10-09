"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const pages = [
  { href: "/", label: "Movie Agent" },
  { href: "/manual", label: "Manual Recommend" },
  { href: "/about", label: "About" },
];

export default function Navigation() {
  const pathname = usePathname();
  return (
    <nav className="siteNavigation" aria-label="Main navigation">
      <div className="navigationLinks">
        {pages.map(({ href, label }) => (
          <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined}>
            {label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
