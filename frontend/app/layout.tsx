import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI vs Real Image Detection",
  description: "Upload an image and detect whether it is AI generated or real.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="flex min-h-full flex-col antialiased">{children}</body>
    </html>
  );
}
