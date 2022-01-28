#!/usr/bin/env bash

#directory="binutils-2.30"
#directory="coreutils-8.29"
#directory="diffutils-3.1"
#directory="findutils-4.7.0"
#directory="openssl-1.0.1u"
#directory="openssl-1.0.1f"
#directory="busybox-1.32.0"
#directory="curl-7.71.1"
#directory="gmp-6.2.0"
#directory="ImageMagick-7.0.10-27"
#directory="libmicrohttpd-0.9.71"
#directory="libtomcrypt-1.18.2"
#directory="sqlite-3.34.0"
#directory="zlib-1.2.11"
#directory="putty-0.74"
#directory="httpd-2.4.48"
#directory="lynx2.8.9"
#directory="nginx-1.21.1"
directory="openssl-1.0.2h"

opts=(O0 O1 O2 O3)
#archs=(x86-32 x86-64 arm-32 mips-32)
archs=(arm-32)
#archs=(x86-32 x86-64)
#archs=(arm-32 arm-64)
#archs=(mips-32 mips-64)

#export CC="gcc"

for opt in ${opts[*]}; do
  for arch in ${archs[*]}; do

    tar -xf "$directory".tar.gz
    #    tar -xf "$directory".tar.xz
    #    tar -xf "$directory".tar.bz2

    cd "$directory"

    #nginx
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure --with-cc-opt="-$opt" LDFLAGS=-m32 TIME_T_32_BIT_OK=yes --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure --with-cc-opt="-$opt" --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure --with-cc-opt="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    else
    #      ./configure --with-cc-opt="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-none-linux-gnueabi --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    fi
    #    make -j8
    #    make install

    #lynx
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CFLAGS="-$opt -m32" LDFLAGS=-m32 TIME_T_32_BIT_OK=yes --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CFLAGS="-$opt" --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure CFLAGS="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    else
    #      ./configure CFLAGS="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-none-linux-gnueabi --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    fi
    #    make -j8
    #    make install

    #httpd
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CFLAGS="-$opt -m32" LDFLAGS=-m32 TIME_T_32_BIT_OK=yes --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CFLAGS="-$opt" --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure CFLAGS="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    else
    #      ./configure CFLAGS="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-none-linux-gnueabi --prefix=/media/onetb/git_repos/trex-release/data-raw/"$directory"/install
    #    fi
    #    make -j8
    #    make install

    # findutils
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CFLAGS="-$opt -m32" LDFLAGS=-m32 TIME_T_32_BIT_OK=yes --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure CFLAGS="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    else
    #      ./configure CFLAGS="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-none-linux-gnueabi --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    fi
    #    make -j8
    #    make install

    # busybox
    # x86/x64
    #    make defconfig
    #    if [[ $arch == "x86-32" ]]; then
    #      EXTRA_CFLAGS="-$opt -m32" EXTRA_LDFLAGS='-m32' make -j4
    #    else
    #      EXTRA_CFLAGS="-$opt" make -j4
    #    fi
    # ARM vs. MIPS
    #    if [[ $arch == "arm-32" ]]; then
    #      make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi- defconfig
    #      EXTRA_CFLAGS="-$opt" make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi- -j4
    #    else
    #      make ARCH=mips CROSS_COMPILE=mips-linux-gnu- defconfig
    #      EXTRA_CFLAGS="-$opt" make ARCH=mips CROSS_COMPILE=mips-linux-gnu- -j4
    #    fi

    # curl
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CFLAGS="-$opt -m32" --without-zlib --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CFLAGS="-$opt" --without-zlib --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure CFLAGS="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu --without-zlib --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    else
    #      ./configure CFLAGS="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-none-linux-gnueabi --without-zlib --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    fi
    #    make -j6
    #    make install

    # gmp
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CFLAGS="-$opt -m32" ABI=32
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CFLAGS="-$opt"
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure CFLAGS="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu
    #    else
    #      ./configure CFLAGS="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-none-linux-gnueabi
    #    fi
    #    make -j8

    # imagemagic
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CFLAGS="-$opt -m32" CXXFLAGS="-$opt -m32" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install --without-png --without-fontconfig --without-freetype --without-zlib
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install --without-png --without-fontconfig --without-freetype --without-zlib
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure CFLAGS="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install --without-png --without-fontconfig --without-freetype --without-zlib
    #    else
    #      ./configure CFLAGS="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-linux-gnueabi --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install --without-png --without-fontconfig --without-freetype --without-zlib
    #    fi
    #    make -j8
    #    make install

    # libmicrohttpd
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CC=gcc CFLAGS="-$opt -m32" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CC=gcc CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure --host=mips-linux-gnu CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    else
    #      ./configure --host=arm-linux-gnueabi CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    fi
    #    make -j8
    #    make install

    # libtomcrypt
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      make CC=gcc CFLAGS="-$opt -m32" -j8
    #      make install PREFIX=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      make CC=gcc CFLAGS="-$opt" -j8
    #      make install PREFIX=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      make CC=mips-linux-gnu-gcc CFLAGS="-$opt" -j8
    #      make install PREFIX=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    else
    #      make CC=arm-linux-gnueabi-gcc CFLAGS="-$opt" -j8
    #      make install PREFIX=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    fi

    # sqlite
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CFLAGS="-$opt -m32" LDFLAGS="-m32 -L/usr/lib/i386-linux-gnu/" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure CFLAGS="-$opt" CC=mips-linux-gnu-gcc --host=mips-linux-gnu --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    else
    #      ./configure CFLAGS="-$opt" CC=arm-linux-gnueabi-gcc --host=arm-none-linux-gnueabi --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    fi
    #    make -j8
    #    make install

    # zlib
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install CC=gcc CFLAGS="-$opt -m32" ./configure
    #    elif [[ $arch == "x86-64" ]]; then
    #      prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install CC=gcc CFLAGS="-$opt" ./configure
    #    elif [[ $arch == "mips-32" ]]; then
    #      prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install CC=mips-linux-gnu-gcc CFLAGS="-$opt" ./configure
    #    else
    #      prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install CC=arm-linux-gnueabi-gcc CFLAGS="-$opt" ./configure
    #    fi
    #    make -j8
    #    make install

    # puttygen
    #    mkdir install
    #    if [[ $arch == "x86-32" ]]; then
    #      ./configure CC=gcc CFLAGS="-$opt -m32" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "x86-64" ]]; then
    #      ./configure CC=gcc CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    elif [[ $arch == "mips-32" ]]; then
    #      ./configure --host=mips-linux-gnu CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    else
    #      ./configure --host=arm-linux-gnueabi CFLAGS="-$opt" --prefix=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    fi
    #    make -j8
    #    make install

    # x86 openssl
    #    mkdir /media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    sed -i "358 s/-O3/-m32/" ./Configure #1.0.1u
    ##    sed -i "348 s/-O3/-m32/" ./Configure #1.0.1f
    #    ./Configure -"$opt" linux-generic32 --openssldir=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    make -j8
    #    make test
    #    make install_sw

    # x64 openssl
    #    mkdir /media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    ##    sed -i "373 s/-O3/-$opt/" ./Configure #1.0.1u
    #    sed -i "363 s/-O3/-$opt/" ./Configure #1.0.1f
    #    ./Configure -"$opt" linux-x86_64 --openssldir=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    make -j8
    #    make test
    #    make install_sw

    # arm
    #    export CFLAGS="-$opt"
    #    if [[ $arch == arm32 ]]; then
    #      ./configure --host=arm-linux-gnueabi
    #    else
    #      ./configure --host=aarch64-linux-gnu
    #
    #      # coreutils problem
    #      #      sed -i 's/SYS_getdents/SYS_getdents64/g' src/ls.c
    #    fi

    # arm openssl
#    mkdir /media/onetb/git_repos/trex-release/data-raw/case/raw_query/"$directory"/install
    #    sed -i "365 s/-O3 //" ./Configure #1.0.1u
    sed -i "416 s/-O3 //" ./Configure #1.0.2h
    #    #    sed -i "355 s/-O3 //" ./Configure #1.0.1f
    ./Configure linux-elf no-asm -shared -"$opt" --cross-compile-prefix=arm-linux-gnueabi-
#    ./Configure linux-elf no-asm -shared -"$opt" --cross-compile-prefix=arm-linux-gnueabi- --openssldir=/media/onetb/git_repos/trex-release/data-raw/case/raw_query/"$directory"/install
    make depend
    make -j8
#    make test
#    make install_sw

    # mips
    #    export CFLAGS="-$opt "
    #    if [[ $arch == mips32 ]]; then
    #      ./configure --host=mips-linux-gnu
    #    else
    #      ./configure --host=mips64-linux-gnuabi64
    #
    #      # coreutils problem
    #      #      sed -i 's/SYS_getdents/SYS_getdents64/g' src/ls.c
    #    fi

    # mips openssl
    #    mkdir /media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    #    sed -i "365 s/-O3 //" ./Configure #1.0.1u
    #    sed -i "355 s/-O3 //" ./Configure #1.0.1f
    #    ./Configure linux-elf no-asm -"$opt" --cross-compile-prefix=mips-linux-gnu- --openssldir=/media/onetb/git_repos/bin2vec/data-raw/"$directory"/install
    #    make -j4
    #    make test
    #    make install_sw

    # remove diff-3.2 fgets error ("security hole...)"
    #    sed -i "1012 s/^/\/\//" lib/stdio.h
    #    make -j4

    # remove fgets error ("security hole...)"
    #sed -i "1030 s/^/\/\//" lib/stdio.h
    #make -j4

    mkdir -p ../bin/"$arch"/"$directory"-"$opt"

    # coreutils and diffutils
    #find ./src -executable -exec file {} \; | grep -i elf | cut -d: -f1 | xargs cp -t ../bin/"$arch"/"$directory"-"$opt"/

    # binutils
    #    find ./binutils -executable -exec file {} \; | grep -i elf | cut -d: -f1 | xargs cp -t ../bin/"$arch"/"$directory"-"$opt"/

    # findutils
    #    cp install/bin/* ../bin/"$arch"/"$directory"-"$opt"/

    # findutils
#    cp install/sbin/* ../bin/"$arch"/"$directory"-"$opt"/

    # httpd
    #    cp install/bin/* ../bin/"$arch"/"$directory"-"$opt"/
    #    cp install/modules/* ../bin/"$arch"/"$directory"-"$opt"/

    # lynx
    #    cp install/bin/* ../bin/"$arch"/"$directory"-"$opt"/

    # openssl
    cp ./install/bin/openssl ../bin/"$arch"/"$directory"-"$opt"/

    # busybox
    #    cp ./busybox ../bin/"$arch"/"$directory"-"$opt"/

    # curl
    #    cp ./install/lib/libcurl.so.4.6.0 ../bin/"$arch"/"$directory"-"$opt"/

    # gmp
    #    cp .libs/libgmp.so.10.4.0 ../bin/"$arch"/"$directory"-"$opt"/

    # imagemagic
    #    cp install/lib/libMagick++-7.Q16HDRI.so.4.0.0 ../bin/"$arch"/"$directory"-"$opt"/
    #    cp install/lib/libMagickCore-7.Q16HDRI.so.7.0.0 ../bin/"$arch"/"$directory"-"$opt"/
    #    cp install/lib/libMagickWand-7.Q16HDRI.so.7.0.0 ../bin/"$arch"/"$directory"-"$opt"/

    # libmicrohttpd
    #    cp install/lib/libmicrohttpd.a ../bin/"$arch"/"$directory"-"$opt"/

    # libtomcrypt
    #    cp install/lib/libtomcrypt.a ../bin/"$arch"/"$directory"-"$opt"/

    # sqlite
    #    cp install/lib/libsqlite3.so.0.8.6 ../bin/"$arch"/"$directory"-"$opt"/

    # zlib
    #    cp install/lib/libz.so.1.2.11 ../bin/"$arch"/"$directory"-"$opt"/

    # putty
    #    cp install/bin/* ../bin/"$arch"/"$directory"-"$opt"/

    cd ..
    rm -rf "$directory"

  done
done
