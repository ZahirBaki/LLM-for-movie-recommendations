import { notFound } from "next/navigation";
import EvaluationReview from "./EvaluationReview";

export const metadata = { title: "Local recommendation review", robots: { index: false, follow: false } };

export default function EvaluationPage() {
  if (process.env.NODE_ENV !== "development") notFound();
  return <EvaluationReview />;
}
