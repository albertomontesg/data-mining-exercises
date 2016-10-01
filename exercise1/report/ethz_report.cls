\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ethz_report}[2016/10/01 v0.1 My report template]

\LoadClass[10pt, a4paper, twoside]{article}
\RequirePackage{graphicx}
\RequirePackage[english]{babel}
\RequirePackage{amsmath,amssymb,amsfonts,textcomp}
\RequirePackage{float,flafter}
\RequirePackage{hyperref}
\RequirePackage{inputenc}
\RequirePackage[a4paper, left=2.5cm, right=2.5cm, top=2cm, bottom=3cm]{geometry}

\def\subject#1{\def\@subject{#1}}
\def\email#1{\def\@email{#1}}

\renewcommand\maketitle{
\hrule
\begin{center}
    \vspace{.4cm}
    \bf{\Huge \@title} \\
    \vspace{.3cm}
    \large \@subject
    \vspace{.2cm}
\end{center}
{\bf \@author} (\@email) \hspace{\fill} \@date \\
\hrule

}
\endinput
