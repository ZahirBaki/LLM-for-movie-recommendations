"use client";

import { createContext, useContext, useState, type ReactNode, type Dispatch, type SetStateAction } from "react";
import { usePathname } from "next/navigation";
import Navigation from "./Navigation";
import MovieChat from "./MovieChat";

const MovieRegion = createContext<{ countryCode: string; setCountryCode: Dispatch<SetStateAction<string>> } | null>(null);

export function useMovieRegion() {
  const region = useContext(MovieRegion);
  if (!region) throw new Error("MovieRegion requires MovieApp");
  return region;
}

export default function MovieApp({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [countryCode, setCountryCode] = useState("SE");
  return (
    <MovieRegion.Provider value={{ countryCode, setCountryCode }}>
      <div className={pathname === "/" ? "movieApp agentApp" : "movieApp"}>
        <Navigation />
        {/* Keep the existing chat and its request/session refs alive across Link navigation.
            hidden removes it from display, keyboard navigation, and the accessibility tree. */}
        <main className="page agentPage" hidden={pathname !== "/"}>
          <h1 className="agentTitle">Movie Agent</h1>
          <MovieChat apiUrl="http://127.0.0.1:8000" countryCode={countryCode} />
        </main>
        {children}
      </div>
    </MovieRegion.Provider>
  );
}
