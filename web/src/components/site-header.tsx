"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Map, Satellite, Activity, Terminal, Images } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/dataset", label: "Your xBD", icon: Images },
  { href: "/map", label: "Damage map", icon: Map },
  { href: "/model", label: "ResNet50", icon: Satellite },
  { href: "/run", label: "Run on xBD", icon: Terminal },
];

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-[#0b1220]/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3">
        <Link href="/" className="flex items-center gap-3">
          <span className="flex size-9 items-center justify-center rounded-lg bg-orange-500/20 text-orange-300 ring-1 ring-orange-400/30">
            <Satellite className="size-4" />
          </span>
          <span>
            <span className="block text-sm font-semibold tracking-tight text-white">
              DamageMap
            </span>
            <span className="block text-[11px] text-slate-400">
              xBD · Joint Damage Scale · ResNet50
            </span>
          </span>
        </Link>
        <nav className="flex flex-wrap items-center gap-1">
          {links.map((link) => {
            const active =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);
            const Icon = link.icon;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm",
                  active
                    ? "bg-white/10 text-white"
                    : "text-slate-400 hover:bg-white/5 hover:text-white"
                )}
              >
                <Icon className="size-3.5" />
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
